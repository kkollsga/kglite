//! reStructuredText extraction for the `code_tree` docs pass.
//!
//! Sphinx/`.rst` is the dominant documentation toolchain for the scientific-
//! Python ecosystem (numpy, pandas, xarray, scipy, …), so capturing it is the
//! biggest lever for doc↔code linking on those repos. RST is *richer* than
//! Markdown for our purpose: cross-reference **roles** name symbols explicitly,
//! e.g. ``:func:`~xarray.open_dataset` `` / ``:class:`Dataset` `` — no heuristic
//! needed, the author told us it's a symbol.
//!
//! Extracted here (everything else downstream is shared with the Markdown path):
//! - **title** — the first section heading (text underlined by an adornment run).
//! - **headings** — every section heading, in document order.
//! - **mention candidates** — Python-domain roles (``:func:`` / ``:class:`` /
//!   ``:meth:`` / …) and double-backtick inline literals (RST's code span).
//! - **links** — ``:doc:`path` `` cross-references → doc→doc edges.

use super::{ident_path_re, resolve_rel_path, strip_doc_ext, Candidate, DocEntry, DocFormat};
use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

/// Adornment characters that can underline (or overline) an RST section title.
const ADORNMENT: &str = "=-`:'\"~^_*+#<>.";

/// Leaf role names that name a code symbol (Python domain + defaults). The
/// domain prefix (`py:func`) is stripped before this check.
const CODE_ROLES: &[&str] = &[
    "func", "function", "meth", "method", "class", "obj", "attr", "data", "exc", "const", "member",
    "type",
];

/// Parse one `.rst` file into a [`DocEntry`]. Returns `None` on read error.
pub(super) fn parse(rel_path: &str, abs_path: &Path) -> Option<DocEntry> {
    let body = std::fs::read_to_string(abs_path).ok()?;
    let concept_id = strip_doc_ext(rel_path).to_string();
    let title = first_section(&body).unwrap_or_else(|| {
        concept_id
            .rsplit('/')
            .next()
            .unwrap_or(&concept_id)
            .to_string()
    });
    Some(DocEntry {
        concept_id,
        file_path: rel_path.to_string(),
        title,
        body,
        props: Vec::new(),
        format: DocFormat::Rst,
    })
}

/// All section titles in document order, capped at [`super::MAX_HEADINGS`].
pub(super) fn headings(body: &str) -> Vec<String> {
    let mut out = sections(body);
    out.truncate(super::MAX_HEADINGS);
    out
}

/// The first section title, if any.
fn first_section(body: &str) -> Option<String> {
    sections(body).into_iter().next()
}

/// Section titles: a non-indented text line immediately followed by an
/// adornment run (a line of a single repeated punctuation char) at least as long
/// as the title. The overline+underline form is handled naturally — the overline
/// is itself an adornment line and so never matches as a title.
fn sections(body: &str) -> Vec<String> {
    let lines: Vec<&str> = body.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < lines.len() {
        let text = lines[i];
        let title = text.trim();
        let is_title_line = !title.is_empty()
            && !text.starts_with([' ', '\t'])
            && title.chars().count() <= 200
            && adornment_char(text).is_none();
        if is_title_line {
            if let Some(_c) = adornment_char(lines[i + 1]) {
                if lines[i + 1].trim_end().chars().count() >= title.chars().count() {
                    out.push(title.to_string());
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// `Some(ch)` if `line` is an adornment run (≥2 of a single punctuation char
/// from [`ADORNMENT`], nothing else).
fn adornment_char(line: &str) -> Option<char> {
    let t = line.trim_end();
    let mut chars = t.chars();
    let first = chars.next()?;
    if !ADORNMENT.contains(first) || t.chars().count() < 2 {
        return None;
    }
    t.chars().all(|c| c == first).then_some(first)
}

/// `:role:`content` ` — role name (possibly `domain:role`) + backtick content.
fn role_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r":([a-zA-Z][a-zA-Z0-9_:+-]*):`([^`]+)`").unwrap())
}

/// ``` ``literal``` ``` — RST inline literal (a code span).
fn literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"``([^`]+)``").unwrap())
}

/// `:doc:`target` ` — a Sphinx doc cross-reference.
fn doc_role_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r":doc:`([^`]+)`").unwrap())
}

/// Pull the symbol target out of a role's content: unwrap the `text <target>`
/// form, drop a leading `~` (Sphinx "show last component"), and normalize to the
/// leading identifier path (strips `()` and trailing punctuation).
fn role_target(content: &str) -> Option<String> {
    let inner = match (content.rfind('<'), content.rfind('>')) {
        (Some(a), Some(b)) if b > a => &content[a + 1..b],
        _ => content,
    };
    let inner = inner.trim().trim_start_matches('~').trim();
    ident_path_re().find(inner).map(|m| m.as_str().to_string())
}

/// RST mention candidates: Python-domain roles + double-backtick literals. Both
/// are strong code signals → the unique-bare-name fallback is allowed.
pub(super) fn candidates(body: &str) -> Vec<Candidate> {
    let mut out = Vec::new();
    for cap in role_re().captures_iter(body) {
        let role = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let leaf = role.rsplit(':').next().unwrap_or(role).to_ascii_lowercase();
        if !CODE_ROLES.contains(&leaf.as_str()) {
            continue;
        }
        if let Some(token) = cap.get(2).and_then(|m| role_target(m.as_str())) {
            out.push(Candidate {
                token,
                allow_fallback: true,
            });
        }
    }
    for cap in literal_re().captures_iter(body) {
        let span = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        if let Some(m) = ident_path_re().find(span) {
            out.push(Candidate {
                token: m.as_str().to_string(),
                allow_fallback: true,
            });
        }
    }
    out
}

/// RST outbound links: `:doc:`path` ` cross-references → doc targets (resolved
/// relative to the linking doc's directory; a leading `/` is repo-root-absolute).
pub(super) fn link_targets(body: &str, src_dir: &str) -> Vec<super::LinkTarget> {
    let mut out = Vec::new();
    for cap in doc_role_re().captures_iter(body) {
        let content = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        // `:doc:`text <path>`` keeps the path; else the content is the path.
        let raw = match (content.rfind('<'), content.rfind('>')) {
            (Some(a), Some(b)) if b > a => &content[a + 1..b],
            _ => content,
        };
        if let Some(target) = resolve_rel_path(raw.trim(), src_dir) {
            out.push(super::LinkTarget::Doc(target));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_title_and_outline() {
        let body = "Top Title\n=========\n\nIntro.\n\nSubsection\n----------\nbody\n";
        assert_eq!(first_section(body).as_deref(), Some("Top Title"));
        assert_eq!(headings(body), vec!["Top Title", "Subsection"]);
    }

    #[test]
    fn overline_underline_section_counts_once() {
        let body = "======\nTitle\n======\n\ntext\n";
        assert_eq!(headings(body), vec!["Title"]);
    }

    #[test]
    fn roles_extract_symbol_targets() {
        let toks: Vec<String> = candidates(
            "Use :func:`~xarray.open_dataset` and :class:`Dataset` and \
             :meth:`Dataset.mean` plus ``literal_code`` here.",
        )
        .into_iter()
        .map(|c| c.token)
        .collect();
        assert!(toks.contains(&"xarray.open_dataset".to_string()));
        assert!(toks.contains(&"Dataset".to_string()));
        assert!(toks.contains(&"Dataset.mean".to_string()));
        assert!(toks.contains(&"literal_code".to_string()));
    }

    #[test]
    fn non_code_roles_ignored() {
        // :ref: / :term: are not code roles.
        let toks: Vec<String> = candidates(":ref:`somewhere` and :term:`glossary`")
            .into_iter()
            .map(|c| c.token)
            .collect();
        assert!(toks.is_empty());
    }

    #[test]
    fn doc_refs_resolve_relative_and_absolute() {
        let rel: Vec<String> = link_targets(":doc:`io` and :doc:`/user-guide/plotting`", "doc")
            .into_iter()
            .map(|t| match t {
                super::super::LinkTarget::Doc(s) => s,
                super::super::LinkTarget::File(s) => s,
            })
            .collect();
        assert_eq!(rel, vec!["doc/io", "user-guide/plotting"]);
    }
}
