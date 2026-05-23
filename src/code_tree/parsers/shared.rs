//! Shared parser helpers (ported from parsers/base.py).

use crate::code_tree::models::Annotation;
use aho_corasick::AhoCorasick;
use regex::Regex;
use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;
use tree_sitter::Node;

/// Build a module path from a file's location, joining the source-root
/// directory name and the file stem with `separator` (`.` for Dart /
/// Swift / HTML / CSS / JS / similar; `\` for PHP namespace style).
///
/// Phase A.3 / 0.9.53 code-rot cleanup — consolidated from per-parser
/// copies in `dart.rs`, `html.rs`, `php.rs`, `swift.rs`, `css.rs`.
pub(super) fn file_to_module_path(filepath: &Path, src_root: &Path, separator: char) -> String {
    let stem = filepath.file_stem().and_then(|o| o.to_str()).unwrap_or("");
    let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
    match (pkg.is_empty(), stem.is_empty()) {
        (true, _) => stem.to_string(),
        (false, true) => pkg.to_string(),
        (false, false) => format!("{pkg}{separator}{stem}"),
    }
}

/// Combine module path / owner prefix / name into a single qualified
/// identifier, joined with `separator` (`.` for most languages, `\` for
/// PHP). When an `owner_prefix` is present (e.g. an enclosing class /
/// struct / namespace), it wins over the module path — the owner is the
/// closer scope.
///
/// Phase A.3 / 0.9.53 code-rot cleanup — consolidated from
/// `dart.rs`, `php.rs`, `swift.rs`.
pub(super) fn make_qualified(
    module_path: &str,
    owner_prefix: &str,
    name: &str,
    separator: char,
) -> String {
    match (module_path.is_empty(), owner_prefix.is_empty()) {
        (true, true) => name.to_string(),
        (true, false) => format!("{owner_prefix}{separator}{name}"),
        (false, true) => format!("{module_path}{separator}{name}"),
        (false, false) => format!("{owner_prefix}{separator}{name}"),
    }
}

/// Fast byte-level filter: do any of the annotation keywords appear in
/// `source` at all? Built with aho-corasick (case-insensitive, no word
/// boundaries — accuracy is the regex's job; this is just a "skip the
/// AST walk entirely" precheck). Returns true on the first match.
fn has_annotation_keyword(source: &[u8]) -> bool {
    static AC: OnceLock<AhoCorasick> = OnceLock::new();
    let ac = AC.get_or_init(|| {
        AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build([
                "TODO", "FIXME", "HACK", "SAFETY", "XXX", "BUG", "NOTE", "WARNING",
            ])
            .expect("aho-corasick patterns compile")
    });
    ac.is_match(source)
}

/// Strict test-file detection shared across language parsers.
///
/// Returns true when the file is plausibly a test file. Two signals:
/// - `filename` ends with one of the language-specific `suffix_patterns`
///   (e.g. `.test.ts`, `Tests.swift`, `Test.php`).
/// - Any segment of `rel_path` is a conventional test directory:
///   `test`, `tests`, `__tests__`, `spec`, `specs`.
///
/// Path-segment-aware so we don't false-positive on names like
/// `latest.html`, `contest.css`, or `protest.swift` — the previous
/// `rel_path.to_lowercase().contains("test")` check did. Standard Go
/// (`*_test.go`) and Python (`test_*.py` / `*_test.py`) detection lives
/// inside their respective parsers because the conventions are specific
/// enough that this helper isn't a good fit.
pub fn is_test_path(rel_path: &str, filename: &str, suffix_patterns: &[&str]) -> bool {
    if suffix_patterns.iter().any(|s| filename.ends_with(s)) {
        return true;
    }
    let normalised = rel_path.replace('\\', "/");
    for segment in normalised.split('/') {
        if matches!(segment, "test" | "tests" | "__tests__" | "spec" | "specs") {
            return true;
        }
    }
    false
}

/// Extract the UTF-8 text of a tree-sitter node.
#[inline]
pub fn node_text<'a>(node: Node<'a>, source: &'a [u8]) -> &'a str {
    // UTF-8 boundaries are enforced by tree-sitter; safe unwrap on well-formed input.
    std::str::from_utf8(&source[node.byte_range()]).unwrap_or("")
}

/// Count lines of code in `source`. Matches the Python `count_lines` semantics:
/// number of newlines, plus one if the file is non-empty and doesn't end with `\n`.
pub fn count_lines(source: &[u8]) -> u32 {
    let newlines = bytecount_newlines(source);
    if !source.is_empty() && !source.ends_with(b"\n") {
        newlines + 1
    } else {
        newlines
    }
}

#[inline]
fn bytecount_newlines(src: &[u8]) -> u32 {
    src.iter().filter(|&&b| b == b'\n').count() as u32
}

/// Extract generic/template parameters from a declaration node.
///
/// Looks for a child whose type matches `node_type` (default: `"type_parameters"`)
/// and returns the inner text with surrounding `<>` or `[]` stripped.
pub fn get_type_parameters(node: Node, source: &[u8], node_type: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == node_type {
            let text = node_text(child, source).trim();
            let stripped = if let Some(inner) =
                text.strip_prefix('<').and_then(|s| s.strip_suffix('>'))
            {
                inner.trim()
            } else if let Some(inner) = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                inner.trim()
            } else {
                text
            };
            if stripped.is_empty() {
                return None;
            }
            return Some(stripped.to_string());
        }
    }
    None
}

fn annotation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(TODO|FIXME|HACK|SAFETY|XXX|BUG|NOTE|WARNING)\b[:\s]*(.*)")
            .expect("annotation regex compiles")
    })
}

/// Default comment node kinds scanned by `extract_comment_annotations`.
pub const DEFAULT_COMMENT_TYPES: &[&str] = &["line_comment", "block_comment", "comment"];

/// Scan every comment node under `root` for TODO/FIXME/etc annotations.
///
/// Iterative stack traversal — avoids blowing Rust's call stack on deeply
/// nested ASTs (same reason the Python version uses an explicit stack).
///
/// Hot-path optimisation: the vast majority of files in real codebases
/// have no annotation keywords at all. A 64-bit byte-pattern pre-check
/// over the raw source skips the full AST walk in that case (which on
/// dotnet/runtime drops ~30% of the C# parse phase wall time).
pub fn extract_comment_annotations(
    root: Node,
    source: &[u8],
    comment_types: &[&str],
) -> Option<Vec<Annotation>> {
    if !has_annotation_keyword(source) {
        return None;
    }
    let re = annotation_regex();
    let mut out: Vec<Annotation> = Vec::new();
    let mut stack: Vec<Node> = vec![root];

    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if comment_types.iter().any(|t| *t == kind) {
            let text = node_text(node, source);
            for caps in re.captures_iter(text) {
                let kind = caps
                    .get(1)
                    .map(|m| m.as_str().to_ascii_uppercase())
                    .unwrap_or_default();
                let body = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
                // Cap at 200 chars — `&body[..200]` would panic when byte
                // 200 lands inside a multi-byte char (e.g. a `─` rule in a
                // comment), so truncate on a char boundary.
                let truncated: String = body.chars().take(200).collect();
                out.push(Annotation {
                    kind,
                    text: truncated,
                    line: node.start_position().row as u32 + 1,
                });
            }
        }
        // Push children in reverse so we pop them in source order.
        let mut cursor = node.walk();
        let children: Vec<Node> = node.children(&mut cursor).collect();
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// ── Complexity counters (cyclomatic-style) ────────────────────────────

/// Branch-node kinds for each language's tree-sitter grammar.
///
/// These names are checked against `node.kind()` while walking a function body
/// to compute `branch_count` and `max_nesting`. Each entry adds 1 to the branch
/// count *and* contributes to nesting depth — so deeply nested `if`s and `for`s
/// surface a meaningful `max_nesting` value, while flat boolean chains
/// (`a && b && c`) only inflate `branch_count`.
///
/// Lists deliberately include short-circuit / ternary forms so the count
/// approximates McCabe cyclomatic complexity rather than just structural nesting.
pub const BRANCH_KINDS_PYTHON: &[&str] = &[
    "if_statement",
    "elif_clause",
    "while_statement",
    "for_statement",
    "except_clause",
    "case_clause",
    "conditional_expression", // ternary  a if c else b
    "boolean_operator",       // and / or
];

pub const BRANCH_KINDS_RUST: &[&str] = &[
    "if_expression",
    "while_expression",
    "while_let_expression",
    "for_expression",
    "loop_expression",
    "match_arm",
    "try_expression", // the `?` operator
];

pub const BRANCH_KINDS_GO: &[&str] = &[
    "if_statement",
    "for_statement",
    "expression_case",
    "type_case",
    "communication_case",
];

pub const BRANCH_KINDS_JAVA: &[&str] = &[
    "if_statement",
    "while_statement",
    "for_statement",
    "enhanced_for_statement",
    "do_statement",
    "switch_label",
    "catch_clause",
    "ternary_expression",
];

pub const BRANCH_KINDS_TS: &[&str] = &[
    "if_statement",
    "while_statement",
    "for_statement",
    "for_in_statement",
    "for_of_statement",
    "do_statement",
    "switch_case",
    "catch_clause",
    "ternary_expression",
];

pub const BRANCH_KINDS_CPP: &[&str] = &[
    "if_statement",
    "while_statement",
    "for_statement",
    "for_range_loop",
    "do_statement",
    "case_statement",
    "catch_clause",
    "conditional_expression",
];

pub const BRANCH_KINDS_CSHARP: &[&str] = &[
    "if_statement",
    "while_statement",
    "for_statement",
    "for_each_statement",
    "do_statement",
    "switch_section",
    "catch_clause",
    "conditional_expression",
];

pub const BRANCH_KINDS_DART: &[&str] = &[
    "if_statement",
    "while_statement",
    "for_statement",
    "do_statement",
    "switch_statement_case",
    "switch_expression_case",
    "catch_clause",
    "conditional_expression",
    "if_null_expression",
    "logical_and_expression",
    "logical_or_expression",
    "if_element",
    "for_element",
];

/// Walk `body` and return (branch_count, max_nesting).
///
/// `branch_count` increments for every node whose kind is in `branch_kinds`.
/// `max_nesting` is the deepest stack of nested `branch_kinds` nodes seen.
/// `nested_scope_kinds` (e.g. `function_definition`, `closure_expression`)
/// are NOT descended into — they belong to the caller's nested function/closure
/// and would inflate the outer function's metrics. Pass an empty slice if the
/// caller already filters those before invoking this helper.
pub fn compute_complexity(
    body: Node,
    branch_kinds: &[&str],
    nested_scope_kinds: &[&str],
) -> (u32, u32) {
    let branch_set: HashSet<&str> = branch_kinds.iter().copied().collect();
    let nested_set: HashSet<&str> = nested_scope_kinds.iter().copied().collect();
    let mut count: u32 = 0;
    let mut max_depth: u32 = 0;
    walk(
        body,
        &branch_set,
        &nested_set,
        0,
        &mut count,
        &mut max_depth,
    );
    (count, max_depth)
}

fn walk(
    node: Node,
    branches: &HashSet<&str>,
    nested: &HashSet<&str>,
    depth: u32,
    count: &mut u32,
    max_depth: &mut u32,
) {
    let kind = node.kind();
    let is_branch = branches.contains(kind);
    let next_depth = if is_branch {
        *count += 1;
        let d = depth + 1;
        if d > *max_depth {
            *max_depth = d;
        }
        d
    } else {
        depth
    };
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if nested.contains(child.kind()) {
            continue;
        }
        walk(child, branches, nested, next_depth, count, max_depth);
    }
}

// ── Macro decorator detection (C/C++) ─────────────────────────────────

/// Heuristic: does this token text look like a macro attribute/decorator
/// that the C/C++ parser should ignore when extracting names and types?
///
/// Matches all-uppercase identifiers with optional underscores/digits and
/// length ≥ 2: catches `SPDLOG_INLINE`, `FMT_API`, `FMT_BEGIN_NAMESPACE`,
/// `Q_INVOKABLE`, `BOOST_FOREACH`, `MOZ_ASSERT`, `__attribute__`-style names.
///
/// Excludes single-letter all-caps tokens (`T`, `K`, `N`) so generic template
/// parameter names aren't false-positives.
///
/// Used by `cpp.rs` to skip macro decorators in `get_return_type`, `get_name`,
/// and `extract_parameters` — without this, spdlog-style code like
/// `SPDLOG_INLINE void foo()` parses with `name="unknown"` and
/// `return_type="SPDLOG_INLINE"`.
pub fn looks_like_macro_decorator(text: &str) -> bool {
    if text.len() < 2 {
        return false;
    }
    let mut chars = text.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_uppercase() || first == '_') {
        return false;
    }
    text.chars()
        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

// ── Procedure annotation extraction ───────────────────────────────────

fn procedure_annotation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Recognizes `@procedure: name`, `@procedure name`, `@cypher_procedure: name`
        // when they appear at the start of a line (with optional comment-prefix
        // whitespace). Anchored to line start so prose mentions like
        // "the @procedure: marker tells you..." don't false-positive.
        // Name is `[a-zA-Z_][a-zA-Z0-9_]*` (standard identifier).
        Regex::new(r"(?m)^\s*@(?:cypher_)?procedure[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)")
            .expect("procedure annotation regex compiles")
    })
}

/// Extract all `@procedure: NAME` annotations from a docstring/leading comment
/// block. A single function can register under multiple procedure names via
/// repeated annotations — useful for aliases (e.g. both `betweenness` and
/// `betweenness_centrality` dispatching to the same impl).
///
/// Used to surface project-specific procedure registries (e.g. Cypher CALL
/// procedures, RPC method handlers, CLI subcommand dispatchers) as graph
/// `Procedure` nodes connected to their implementing Function.
pub fn extract_procedure_annotations(docstring: Option<&str>) -> Vec<String> {
    let Some(text) = docstring else {
        return Vec::new();
    };
    procedure_annotation_regex()
        .captures_iter(text)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

// ── Generated / minified file detection ───────────────────────────────

/// Codegen banner markers that — when found in a *comment line within the file's
/// first ~10 non-empty lines* — strongly indicate an auto-generated file.
///
/// Restricted to early comments (not anywhere in the first 2 KiB) so that doc
/// strings or test fixtures that happen to mention the words "auto-generated"
/// or "DO NOT EDIT" deeper in the file don't false-positive.
fn generated_marker_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"(?i)(",
            // protoc, ts-proto, protoc-gen-go banners
            r"\bcode\s+generated\b",
            // Most codegen tools include this exact phrase, often uppercase
            r"|\bdo\s+not\s+edit\b",
            // bindgen, jOOQ, sqlc
            r"|\bautomatically\s+generated\b",
            // C# T4, Visual Studio designer files
            r"|<\s*auto-generated\s*>",
            // JSDoc / Flow / Cargo lockfile / Buck
            r"|@generated\b",
            // OpenAPI generator banner
            r"|\bgenerated\s+by\s+openapi",
            // generic "Generated by <tool>" — restrict to first 5 lines below
            r"|^\s*(?://|#|/\*|\*|--|<!--)\s*Generated\s+by\b",
            r")",
        ))
        .expect("generated marker regex compiles")
    })
}

/// Returns true if the line looks like a comment (starts with a comment
/// delimiter for one of the supported languages).
fn is_comment_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//")
        || t.starts_with('#')
        || t.starts_with("/*")
        || t.starts_with('*')
        || t.starts_with("--")
        || t.starts_with("<!--")
        || t.starts_with(';')
}

/// Detect generated or minified source files by content sniff.
///
/// Returns `Some("generated")` if any of the file's first ~10 non-empty lines
/// is a comment containing a codegen banner marker (e.g. `// Code generated by
/// protoc; DO NOT EDIT.`, `# @generated`, `<auto-generated>`). Restricting the
/// scan to early *comment* lines avoids false-positives on doc strings or test
/// fixtures that quote the same phrases deeper in a hand-written file.
///
/// Returns `Some("minified")` if the file is one extremely long line, or its
/// average non-empty-line width across the first 50 lines exceeds 500 chars
/// (catches webpack/rollup-bundled JS).
///
/// Returns `None` for normal hand-written source.
///
/// Designed to run BEFORE the tree-sitter parse — these files would otherwise
/// inflate the graph with phantom CALLS edges or single-blob "function" nodes
/// that mask real code structure.
pub fn is_generated_or_minified(source: &[u8]) -> Option<&'static str> {
    if source.is_empty() {
        return None;
    }

    // Banner check: scan up to the first 10 non-empty lines, only test the regex
    // against lines that look like comments. Codegen banners always live at
    // the very top of the file inside a comment block.
    let head_len = source.len().min(4096);
    let head_str = std::str::from_utf8(&source[..head_len]).unwrap_or("");
    let re = generated_marker_regex();
    let mut non_empty_seen = 0;
    for line in head_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        non_empty_seen += 1;
        if non_empty_seen > 10 {
            break;
        }
        if is_comment_line(line) && re.is_match(line) {
            return Some("generated");
        }
    }

    // Minified: one huge line, OR average line width above threshold across
    // the first 50 non-empty lines.
    if source.len() >= 1024 {
        let newline_count = source.iter().filter(|&&b| b == b'\n').count();
        if newline_count == 0 {
            return Some("minified");
        }
    }
    let mut lines_seen: u32 = 0;
    let mut total_width: u64 = 0;
    for raw_line in source.split(|&b| b == b'\n') {
        if raw_line.is_empty() {
            continue;
        }
        total_width += raw_line.len() as u64;
        lines_seen += 1;
        if lines_seen >= 50 {
            break;
        }
    }
    if lines_seen >= 5 {
        let avg = total_width / lines_seen as u64;
        if avg > 500 {
            return Some("minified");
        }
    }

    None
}
