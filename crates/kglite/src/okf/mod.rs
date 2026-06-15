//! OKF (Open Knowledge Format) bundle ingestion — read-only, partial.
//!
//! Parses a directory of markdown files with YAML frontmatter, cross-linked by
//! markdown links (Google's OKF, but also Claude memory dirs, skills, and
//! Obsidian vaults), into a [`crate::graph::DirGraph`]. Conceptually `code_tree`
//! for prose knowledge instead of source code.
//!
//! Ingestion is **partial** (like `code_tree`): each concept becomes a node
//! carrying its frontmatter as properties plus a `file_path` pointer; the body
//! is read on demand and is *not* stored unless [`BuildOptions::with_body`] is
//! set. Links become typed edges; dangling link targets become `_provisional`
//! stub nodes. The result is a normal graph — every Cypher feature, algorithm
//! (`CALL leiden`/`pagerank`), and structural rule works over it with no extra
//! surface.
//!
//! This module is gated behind the `okf` Cargo feature (it pulls a YAML parser);
//! the Python wheel enables it, bare builds don't.

pub mod build;
pub mod frontmatter;
pub mod links;
pub mod model;
pub mod walk;

pub use build::build;
pub use model::{BuildOptions, ConceptDoc, Dialect, Link};

use crate::datatypes::values::Value;
use rayon::prelude::*;
use std::path::Path;

/// Read a concept's markdown body on demand (frontmatter stripped). The
/// counterpart to partial ingestion: the graph stores a `file_path` pointer, and
/// this resolves it to the prose when an agent has narrowed to one concept.
/// A file with no frontmatter returns its whole content.
pub fn read_body(path: &Path) -> Result<String, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let (_yaml, body) = frontmatter::split(&text);
    Ok(body)
}

/// Parse a bundle directory into [`ConceptDoc`]s (no graph yet). Files are read
/// and parsed in parallel; a file with malformed frontmatter degrades to a
/// body-only `Concept` rather than being dropped (permissive consumption).
pub fn parse_bundle(root: &Path, opts: &BuildOptions) -> Result<Vec<ConceptDoc>, String> {
    let walked = walk::discover(root, &opts.skip_dirs)?;
    Ok(parse_concepts(&walked.concepts, opts))
}

/// Parse already-discovered concept files into [`ConceptDoc`]s (parallel). Used
/// by [`parse_bundle`], the builder, and `code_tree`'s docs pass (which reuses
/// the OKF parser to ingest a repo's markdown).
pub fn parse_concepts(files: &[walk::DiscoveredFile], opts: &BuildOptions) -> Vec<ConceptDoc> {
    let mut docs: Vec<ConceptDoc> = files
        .par_iter()
        .filter_map(|f| parse_file(f, opts).ok().flatten())
        .collect();
    // Stable order for reproducible builds / tests.
    docs.sort_by(|a, b| a.concept_id.cmp(&b.concept_id));
    docs
}

/// Parse one discovered file into a [`ConceptDoc`]. Returns `Ok(None)` when the
/// file is skipped (no frontmatter while `require_frontmatter` is set).
fn parse_file(f: &walk::DiscoveredFile, opts: &BuildOptions) -> Result<Option<ConceptDoc>, String> {
    let text = std::fs::read_to_string(&f.abs_path)
        .map_err(|e| format!("reading {}: {e}", f.abs_path.display()))?;

    let (yaml, body) = frontmatter::split(&text);
    // Plain markdown (no frontmatter) is skipped by default — the discriminator
    // between structured knowledge (OKF concepts / memories) and normal md.
    if opts.require_frontmatter && yaml.is_none() {
        return Ok(None);
    }

    let concept_id = f
        .rel_path
        .strip_suffix(".md")
        .unwrap_or(&f.rel_path)
        .to_string();

    // Malformed YAML degrades to an empty frontmatter map (the concept still
    // becomes a node — losing the file entirely would be worse).
    let mut fm = frontmatter::parse(&text).unwrap_or_default();

    // Honor the `kg_skip: true` opt-out marker (excludes the file from the sweep).
    if opts.respect_skip && matches!(fm.get(model::SKIP_KEY), Some(Value::Boolean(true))) {
        return Ok(None);
    }

    // Label: top-level `type` → `metadata.type` (Claude memories) → `Concept`.
    let label = fm
        .remove("type")
        .map(value_to_display)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            fm.get("metadata.type")
                .cloned()
                .map(value_to_display)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| model::DEFAULT_LABEL.to_string());

    // Title: `title` → `name` (Claude memories) → first `# H1` heading (so a
    // frontmatter-less README/doc gets its real title, not the file stem) →
    // file stem.
    let title = fm
        .remove("title")
        .map(value_to_display)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            fm.get("name")
                .cloned()
                .map(value_to_display)
                .filter(|s| !s.is_empty())
        })
        .or_else(|| first_heading(&body))
        .unwrap_or_else(|| stem(&concept_id).to_string());

    let props: Vec<(String, Value)> = fm.into_iter().collect();
    let source_dir = parent_dir(&concept_id);
    let links = links::extract_links(&body, source_dir, opts.dialect);
    let body = if opts.with_body { Some(body) } else { None };

    Ok(Some(ConceptDoc {
        concept_id,
        file_path: f.rel_path.clone(),
        label,
        title,
        props,
        links,
        body,
    }))
}

/// Coerce a frontmatter scalar to a display string for label/title use.
fn value_to_display(v: Value) -> String {
    match v {
        Value::String(s) => s,
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        other => format!("{other:?}"),
    }
}

/// First markdown heading (`# ...`) in a body, used as a title fallback for
/// frontmatter-less docs. Skips fenced code blocks; returns the heading text
/// (leading `#`s stripped), or `None`.
fn first_heading(body: &str) -> Option<String> {
    let mut in_fence = false;
    for line in body.lines() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            if let Some(rest) = t.strip_prefix('#') {
                let h = rest.trim_start_matches('#').trim();
                if !h.is_empty() {
                    return Some(h.to_string());
                }
            }
        }
    }
    None
}

/// Last path component of a concept-id (the file stem).
fn stem(concept_id: &str) -> &str {
    concept_id.rsplit('/').next().unwrap_or(concept_id)
}

/// Directory portion of a concept-id (`""` at the bundle root). `pub(crate)` so
/// `code_tree`'s docs pass reuses it to resolve relative markdown links.
pub(crate) fn parent_dir(concept_id: &str) -> &str {
    match concept_id.rfind('/') {
        Some(i) => &concept_id[..i],
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(dir: &Path, rel: &str, content: &str) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, content).unwrap();
    }

    #[test]
    fn parses_concepts_and_skips_reserved() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "tables/orders.md",
            "---\ntype: BigQuery Table\ntitle: Orders\ntags:\n- sales\n---\nPart of [sales](../datasets/sales.md \"PART_OF\").",
        );
        write(
            dir.path(),
            "datasets/sales.md",
            "---\ntype: BigQuery Dataset\n---\nThe sales dataset.",
        );
        write(
            dir.path(),
            "index.md",
            "# Listing\n* [orders](tables/orders.md)",
        );
        write(dir.path(), "notes.txt", "not markdown");

        let docs = parse_bundle(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(docs.len(), 2, "index.md reserved, notes.txt non-md");

        let orders = docs
            .iter()
            .find(|d| d.concept_id == "tables/orders")
            .unwrap();
        assert_eq!(orders.label, "BigQuery Table");
        assert_eq!(orders.title, "Orders");
        assert_eq!(orders.links.len(), 1);
        assert_eq!(orders.links[0].target, "datasets/sales");
        assert_eq!(orders.links[0].conn_type, "PART_OF");

        let sales = docs
            .iter()
            .find(|d| d.concept_id == "datasets/sales")
            .unwrap();
        assert_eq!(sales.label, "BigQuery Dataset");
        assert_eq!(sales.title, "sales", "title falls back to file stem");
    }

    #[test]
    fn no_frontmatter_skipped_by_default_degrades_when_allowed() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "plain.md",
            "# Just a note\n\nNo frontmatter here.",
        );
        // Default: structured-only → plain markdown is skipped.
        let docs = parse_bundle(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(docs.len(), 0);
        // Opt out → it degrades to a Concept.
        let opts = BuildOptions {
            require_frontmatter: false,
            ..BuildOptions::default()
        };
        let docs = parse_bundle(dir.path(), &opts).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].label, "Concept");
        // No frontmatter title/name → falls back to the first H1 heading.
        assert_eq!(docs[0].title, "Just a note");
    }

    #[test]
    fn title_from_first_heading_when_no_frontmatter_fields() {
        let dir = tempdir().unwrap();
        write(dir.path(), "readme.md", "# My Project\n\nIntro text.");
        let opts = BuildOptions {
            require_frontmatter: false,
            ..BuildOptions::default()
        };
        let docs = parse_bundle(dir.path(), &opts).unwrap();
        assert_eq!(docs[0].title, "My Project");
    }

    #[test]
    fn kg_skip_excludes_by_default_and_respects_override() {
        let dir = tempdir().unwrap();
        write(dir.path(), "keep.md", "---\ntype: Note\n---\nkeep me");
        write(
            dir.path(),
            "scratch.md",
            "---\ntype: Note\nkg_skip: true\n---\nignore me",
        );
        // Default: kg_skip files are excluded.
        let docs = parse_bundle(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].concept_id, "keep");
        // respect_skip=false ingests them anyway.
        let opts = BuildOptions {
            respect_skip: false,
            ..BuildOptions::default()
        };
        let docs = parse_bundle(dir.path(), &opts).unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn skip_dirs_prunes_by_name_and_by_path() {
        let dir = tempdir().unwrap();
        write(dir.path(), "keep/a.md", "---\ntype: Note\n---\nkeep");
        write(
            dir.path(),
            "vendor/repos/b.md",
            "---\ntype: Note\n---\nclone",
        );
        write(dir.path(), "deep/cache/c.md", "---\ntype: Note\n---\ndep");

        // bare name matches at any depth; path entry is anchored to the subtree.
        let opts = BuildOptions {
            skip_dirs: vec!["cache".to_string(), "vendor/repos".to_string()],
            ..BuildOptions::default()
        };
        let ids: Vec<String> = parse_bundle(dir.path(), &opts)
            .unwrap()
            .into_iter()
            .map(|d| d.concept_id)
            .collect();
        assert_eq!(ids, vec!["keep/a"]);

        // without skip_dirs all three are ingested.
        let all = parse_bundle(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn label_and_title_fall_back_to_metadata_type_and_name() {
        let dir = tempdir().unwrap();
        // A Claude-memory-shaped file: no top-level `type`/`title`.
        write(
            dir.path(),
            "feedback_x.md",
            "---\nname: Cypher First\nmetadata:\n  type: feedback\n---\nbody",
        );
        let docs = parse_bundle(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].label, "feedback",
            "label falls back to metadata.type"
        );
        assert_eq!(docs[0].title, "Cypher First", "title falls back to name");
    }

    #[test]
    fn with_body_retains_body() {
        let dir = tempdir().unwrap();
        write(dir.path(), "a.md", "---\ntype: Note\n---\nbody content");
        let opts = BuildOptions {
            with_body: true,
            ..BuildOptions::default()
        };
        let docs = parse_bundle(dir.path(), &opts).unwrap();
        assert_eq!(docs[0].body.as_deref(), Some("body content"));
        let docs2 = parse_bundle(dir.path(), &BuildOptions::default()).unwrap();
        assert_eq!(docs2[0].body, None);
    }
}
