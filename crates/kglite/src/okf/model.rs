//! Data model for OKF bundle ingestion.
//!
//! A bundle is a directory tree of markdown files with YAML frontmatter,
//! cross-linked by markdown links. Each non-reserved `.md` file becomes one
//! [`ConceptDoc`]; the links within become edges. The model is deliberately
//! *partial* — the body is not retained unless [`BuildOptions::with_body`] is
//! set, mirroring `code_tree` (store structure + a `file_path` pointer; read the
//! body on demand).

use crate::datatypes::values::Value;

/// Which link / frontmatter conventions to honour when parsing a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// Strict OKF: bundle-relative `[text](/path.md "TYPE")` markdown links.
    /// A missing `type` still degrades to the `Concept` label (OKF mandates
    /// permissive consumption) rather than erroring.
    Okf,
    /// Loose: everything `Okf` does, **plus** Obsidian-style `[[wikilink]]`
    /// resolution (by file stem). For memory dirs / vaults that aren't strict
    /// OKF bundles.
    Loose,
}

impl Dialect {
    /// Parse a dialect name. `None`, `"okf"` → [`Dialect::Okf`]; `"loose"` /
    /// `"obsidian"` → [`Dialect::Loose`]. Unknown strings fall back to `Okf`.
    pub fn parse(name: Option<&str>) -> Self {
        match name.map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("loose") | Some("obsidian") => Dialect::Loose,
            _ => Dialect::Okf,
        }
    }

    /// Whether `[[wikilink]]` syntax is resolved in this dialect.
    pub fn wikilinks(self) -> bool {
        matches!(self, Dialect::Loose)
    }
}

/// Options controlling a bundle build.
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub dialect: Dialect,
    /// Store each concept's markdown body as a `body` property. Off by default
    /// (partial ingestion — read bodies on demand via the file pointer).
    pub with_body: bool,
    /// Reserved for the opt-in embedder pass (stores body vectors for
    /// `text_score`). Not wired in the core loader; honoured by the wheel.
    pub embed: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            dialect: Dialect::Okf,
            with_body: false,
            embed: false,
        }
    }
}

/// A resolved cross-link from a concept to another concept or an external URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// Target concept-id (bundle-relative path minus `.md`) for path links, the
    /// raw wikilink name (resolved in the builder), or the URL for external
    /// links.
    pub target: String,
    /// Edge type from the inference ladder: explicit link title → section
    /// header → `LINKS_TO`.
    pub conn_type: String,
    /// True when `target` is a wikilink awaiting builder resolution.
    pub is_wikilink: bool,
    /// True when `target` is an external `http(s)` URL — becomes a `Source` node
    /// rather than resolving to a concept.
    pub is_external: bool,
}

/// One parsed concept document. Partial by default: `body` is `None` unless
/// `with_body` was requested.
#[derive(Debug, Clone)]
pub struct ConceptDoc {
    /// Bundle-relative path minus `.md`, forward-slashed (e.g. `tables/users`).
    /// Used as the node id and as the link-resolution target.
    pub concept_id: String,
    /// Bundle-relative path to the source file (the on-demand body pointer).
    pub file_path: String,
    /// Node label: frontmatter `type`, or `Concept` when absent.
    pub label: String,
    /// Display title: frontmatter `title`, or the file stem.
    pub title: String,
    /// Flattened frontmatter (excluding `type`/`title`): scalars direct, `tags`
    /// and other sequences as `Value::List`, nested maps flattened to dotted
    /// keys (`metadata.type`).
    pub props: Vec<(String, Value)>,
    /// Resolved outbound links (becoming edges).
    pub links: Vec<Link>,
    /// Body markdown — `Some` only when `with_body` was requested.
    pub body: Option<String>,
}

/// Default edge type when no title or section header gives a more specific one.
pub const DEFAULT_CONN_TYPE: &str = "LINKS_TO";
/// Structural edge type for directory containment (parent dir → child concept).
pub const CONTAINS_CONN_TYPE: &str = "CONTAINS";
/// Node label assigned to concepts with no frontmatter `type`.
pub const DEFAULT_LABEL: &str = "Concept";
/// Node label for synthesized tag nodes; edge type concept → tag.
pub const TAG_LABEL: &str = "Tag";
pub const TAGGED_CONN_TYPE: &str = "TAGGED";
/// Node label for synthesized external-source (URL) nodes.
pub const SOURCE_LABEL: &str = "Source";
