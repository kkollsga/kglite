//! Markdown link extraction with the edge-type inference ladder.
//!
//! A link from concept A to concept B becomes a directed edge. OKF links are
//! untyped (the relationship lives in prose), so the connection type is inferred
//! in three tiers, most-specific first:
//!  1. an explicit link **title** that looks like an edge type
//!     (`[customers](/tables/customers.md "JOINS_WITH")`),
//!  2. the enclosing **section header** (`# Joins` → `JOINS_WITH`,
//!     `# Citations` → `CITES`, …),
//!  3. the generic [`DEFAULT_CONN_TYPE`] (`LINKS_TO`).
//!
//! Links inside fenced code blocks and markdown image links (`![alt](src)`) are
//! ignored. External `http(s)` links are captured as `is_external` (they become
//! `Source` nodes in the builder); `mailto:`, anchors, and non-`.md` directory
//! links are skipped (directory structure is captured separately).

use crate::okf::model::{Dialect, Link, DEFAULT_CONN_TYPE};
use regex::Regex;
use std::sync::OnceLock;

fn link_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // [text](dest) or [text](dest "title"). `text` may not contain ']'.
    RE.get_or_init(|| Regex::new(r#"\[[^\]]*\]\(([^)\s]+)(?:\s+"([^"]*)")?\)"#).unwrap())
}

fn wikilink_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // [[name]] or [[name|alias]].
    RE.get_or_init(|| Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]").unwrap())
}

/// Map a section heading to a connection type, or `None` to fall through.
fn conn_from_heading(heading: &str) -> Option<&'static str> {
    let h = heading.to_ascii_lowercase();
    if h.contains("citation") {
        Some("CITES")
    } else if h.contains("join") {
        Some("JOINS_WITH")
    } else if h.contains("reference") {
        Some("REFERENCES")
    } else if h.contains("related") {
        Some("RELATED")
    } else if h.contains("depend") {
        Some("DEPENDS_ON")
    } else {
        None
    }
}

/// A link title is honoured as an edge type only when it looks like one
/// (`SCREAMING_SNAKE_CASE`) — otherwise it's a human tooltip, not a type.
fn conn_from_title(title: &str) -> Option<String> {
    let t = title.trim();
    if !t.is_empty()
        && t.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && t.chars().next().is_some_and(|c| c.is_ascii_uppercase())
    {
        Some(t.to_string())
    } else {
        None
    }
}

/// Extract resolved outbound links from a concept body.
///
/// `source_dir` is the concept's directory (bundle-relative, `""` at root), used
/// to resolve relative link targets to bundle-relative concept-ids.
pub fn extract_links(body: &str, source_dir: &str, dialect: Dialect) -> Vec<Link> {
    let mut out: Vec<Link> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut in_fence = false;

    for raw in body.lines() {
        let trimmed = raw.trim_start();
        // Toggle fenced code blocks (``` or ~~~).
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            current_heading = Some(rest.trim_start_matches('#').trim().to_string());
            continue;
        }

        let heading_conn = current_heading.as_deref().and_then(conn_from_heading);

        for cap in link_re().captures_iter(raw) {
            let m = cap.get(0).unwrap();
            // Skip markdown image links: `![alt](src)`.
            if m.start() > 0 && raw.as_bytes()[m.start() - 1] == b'!' {
                continue;
            }
            let dest = cap.get(1).map(|d| d.as_str()).unwrap_or("");
            let conn = cap
                .get(2)
                .and_then(|t| conn_from_title(t.as_str()))
                .or_else(|| heading_conn.map(|c| c.to_string()))
                .unwrap_or_else(|| DEFAULT_CONN_TYPE.to_string());
            if is_external_url(dest) {
                // External http(s) link → a Source node (citation / reference).
                push_unique(
                    &mut out,
                    Link {
                        target: dest.to_string(),
                        conn_type: conn,
                        is_wikilink: false,
                        is_external: true,
                    },
                );
            } else if let Some(target) = resolve_target(dest, source_dir) {
                push_unique(
                    &mut out,
                    Link {
                        target,
                        conn_type: conn,
                        is_wikilink: false,
                        is_external: false,
                    },
                );
            }
        }

        if dialect.wikilinks() {
            for cap in wikilink_re().captures_iter(raw) {
                // Strip a `#heading` anchor: `[[Note#Section]]` targets `Note`
                // (mirrors path-link fragment handling). Avoids phantom
                // dangling refs for section links.
                let raw_name = cap.get(1).unwrap().as_str();
                let name = raw_name.split('#').next().unwrap_or(raw_name).trim();
                if name.is_empty() {
                    continue;
                }
                let conn = heading_conn
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| DEFAULT_CONN_TYPE.to_string());
                push_unique(
                    &mut out,
                    Link {
                        // strip a trailing `.md` if the wikilink included it
                        target: name.trim_end_matches(".md").to_string(),
                        conn_type: conn,
                        is_wikilink: true,
                        is_external: false,
                    },
                );
            }
        }
    }
    out
}

fn push_unique(out: &mut Vec<Link>, link: Link) {
    if !out.contains(&link) {
        out.push(link);
    }
}

/// An external link target — `http(s)` only (`mailto:` and other schemes are not
/// turned into Source nodes).
fn is_external_url(dest: &str) -> bool {
    dest.starts_with("http://") || dest.starts_with("https://")
}

/// Resolve a raw markdown link destination to a bundle-relative concept-id, or
/// `None` if it isn't an in-bundle `.md` target (external URL, anchor-only,
/// directory/index link, …).
fn resolve_target(dest: &str, source_dir: &str) -> Option<String> {
    // Drop fragment / query.
    let dest = dest.split(['#', '?']).next().unwrap_or(dest);
    if dest.is_empty() {
        return None;
    }
    // External or non-relative schemes.
    if dest.contains("://") || dest.starts_with("mailto:") {
        return None;
    }
    // Only markdown concepts become edges (directory/index links are handled by
    // structural CONTAINS edges).
    if !dest.ends_with(".md") {
        return None;
    }
    let stem = &dest[..dest.len() - 3]; // strip ".md"

    let normalized = if let Some(abs) = stem.strip_prefix('/') {
        normalize_path_parts(abs.split('/'))
    } else {
        // Relative to the source concept's directory.
        let mut parts: Vec<&str> = if source_dir.is_empty() {
            Vec::new()
        } else {
            source_dir.split('/').collect()
        };
        let combined = parts
            .drain(..)
            .chain(stem.split('/'))
            .collect::<Vec<_>>()
            .join("/");
        normalize_path_parts(combined.split('/'))
    };
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

/// Normalize a path: drop `.`/empty segments, pop on `..`, join with `/`.
fn normalize_path_parts<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    let mut stack: Vec<&str> = Vec::new();
    for p in parts {
        match p {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    stack.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titled_link_yields_typed_edge() {
        let body = "Joined with [customers](/tables/customers.md \"JOINS_WITH\") here.";
        let links = extract_links(body, "tables", Dialect::Okf);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "tables/customers");
        assert_eq!(links[0].conn_type, "JOINS_WITH");
    }

    #[test]
    fn section_header_inference() {
        let body = "# Citations\n[1] [src](/references/x.md)\n# Joins\nsee [y](/tables/y.md)";
        let links = extract_links(body, "tables", Dialect::Okf);
        let by_target: std::collections::HashMap<_, _> = links
            .iter()
            .map(|l| (l.target.as_str(), l.conn_type.as_str()))
            .collect();
        assert_eq!(by_target.get("references/x"), Some(&"CITES"));
        assert_eq!(by_target.get("tables/y"), Some(&"JOINS_WITH"));
    }

    #[test]
    fn untyped_link_defaults_to_links_to() {
        let body = "See [other](./other.md) for details.";
        let links = extract_links(body, "tables", Dialect::Okf);
        assert_eq!(links[0].target, "tables/other");
        assert_eq!(links[0].conn_type, "LINKS_TO");
    }

    #[test]
    fn relative_parent_paths_resolve() {
        let body = "Part of the [sales dataset](../datasets/sales.md).";
        let links = extract_links(body, "tables", Dialect::Okf);
        assert_eq!(links[0].target, "datasets/sales");
    }

    #[test]
    fn tooltip_title_is_not_a_type() {
        let body = "See [customers](/tables/customers.md \"the customers table\").";
        let links = extract_links(body, "tables", Dialect::Okf);
        assert_eq!(links[0].conn_type, "LINKS_TO");
    }

    #[test]
    fn external_captured_non_md_skipped() {
        let body =
            "# Citations\n[site](https://example.com) and [dir](subdir/) and [doc](./pic.png)";
        let links = extract_links(body, "", Dialect::Okf);
        // the http link becomes an external (Source) link; dir/ and .png are skipped
        assert_eq!(links.len(), 1);
        assert!(links[0].is_external);
        assert_eq!(links[0].target, "https://example.com");
        assert_eq!(links[0].conn_type, "CITES");
    }

    #[test]
    fn images_and_fenced_code_skipped() {
        let body =
            "![alt](/tables/x.md)\n```sql\nSELECT [a](/tables/y.md)\n```\n[real](/tables/z.md)";
        let links = extract_links(body, "tables", Dialect::Okf);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "tables/z");
    }

    #[test]
    fn wikilinks_only_in_loose_dialect() {
        let body = "See [[other-note]] and [[sub/thing|alias]].";
        assert!(extract_links(body, "", Dialect::Okf).is_empty());
        let links = extract_links(body, "", Dialect::Loose);
        assert_eq!(links.len(), 2);
        assert!(links[0].is_wikilink);
        assert_eq!(links[0].target, "other-note");
        assert_eq!(links[1].target, "sub/thing");
    }

    #[test]
    fn wikilink_anchor_is_stripped() {
        let links = extract_links(
            "see [[Design Notes#Goals]] and [[api#parse]]",
            "",
            Dialect::Loose,
        );
        let targets: Vec<&str> = links.iter().map(|l| l.target.as_str()).collect();
        assert_eq!(targets, vec!["Design Notes", "api"]);
    }
}
