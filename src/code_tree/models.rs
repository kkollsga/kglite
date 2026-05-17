//! Data models for parsed code entities (ported from parsers/models.py).
//!
//! Field order is preserved from the Python dataclasses so that any
//! downstream JSON (e.g. the `fields` blob embedded on Class/Struct nodes,
//! consumed by the MCP code-review server) remains byte-identical.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

pub type MetadataMap = HashMap<String, serde_json::Value>;

/// An annotation extracted from a comment (TODO/FIXME/HACK/SAFETY/XXX/BUG/NOTE/WARNING).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub kind: String,
    pub text: String,
    pub line: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileInfo {
    /// Path relative to `src_root`.
    pub path: String,
    pub filename: String,
    pub loc: u32,
    /// Language-specific qualified path (e.g. `foo::bar::baz`, `foo.bar.baz`).
    pub module_path: String,
    /// "rust" | "python" | "typescript" | "javascript" | "go" | "java" | "cpp" | "c" | "csharp"
    pub language: String,
    pub submodule_declarations: Vec<String>,
    pub imports: Vec<String>,
    /// Exported names (`__all__`, `pub` items).
    pub exports: Vec<String>,
    /// TODO/FIXME/HACK/SAFETY comments. `None` if none were found.
    pub annotations: Option<Vec<Annotation>>,
    pub is_test: bool,
    /// Set when ingestion deliberately skipped the file. Values:
    /// `"generated"` (auto-generated stubs/codegen), `"minified"` (single-line bundles
    /// or extreme line widths). Skipped files emit no Function/Class/Constant nodes —
    /// only this FileInfo so users can audit what was filtered.
    pub skip_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub qualified_name: String,
    pub visibility: String,
    pub is_async: bool,
    pub is_method: bool,
    pub signature: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub return_type: Option<String>,
    pub decorators: Vec<String>,
    /// `(callee_name, line_number)` pairs.
    pub calls: Vec<(String, u32)>,
    /// Identifier references in the function body that *look like
    /// constants* (terminal segment matches `SCREAMING_SNAKE_CASE`).
    /// The builder resolves these against the project's `ConstantInfo`
    /// set to emit `REFERENCES` edges from Function → Constant.
    /// Stored as `(identifier_text, line_number)` pairs.
    pub references: Vec<(String, u32)>,
    /// Function-pointer references: bare or scoped identifiers passed
    /// as arguments to a higher-order function (e.g.
    /// `iter.and_then(some_fn)`). Distinct from `calls` because the
    /// referenced function isn't necessarily invoked at the reference
    /// site — the value is passed by reference. The builder resolves
    /// these against the project's function set to emit
    /// `REFERENCES_FN` edges. Stored as `(identifier_text, line)` pairs.
    pub function_refs: Vec<(String, u32)>,
    /// Generic/template parameters, e.g. `"T, U: Display"`.
    pub type_parameters: Option<String>,
    pub end_line: Option<u32>,
    /// Structured parameter list — one entry per declared parameter (excluding
    /// `self`/`cls`/`&self`/`&mut self`). Populated by per-language parsers; the
    /// `signature` string remains the user-facing source of truth, this field is
    /// for graph builders that want structured access (e.g. USES_TYPE on params).
    pub parameters: Vec<ParameterInfo>,
    /// Cyclomatic-style branch count over the function body. Counts each
    /// `if`/`elif`/`while`/`for`/`case`/`catch`/ternary/`&&`/`||`-style node the
    /// language grammar exposes. `None` if the parser didn't compute it (e.g. the
    /// function had no body — abstract/protocol methods, `.pyi` stubs).
    pub branch_count: Option<u32>,
    /// Parameter count (excluding `self`/`cls`/`&self`/`&mut self`).
    pub param_count: Option<u32>,
    /// Max nesting depth of branch nodes encountered while walking the body.
    pub max_nesting: Option<u32>,
    /// `Some(true)` if the function body contains a bare-name call resolving to
    /// the function's own short name. Heuristic — refined later by the resolved
    /// CALLS edges.
    pub is_recursive: Option<bool>,
    /// Set when the function's docstring or leading comment contains one or
    /// more `@procedure: NAME` annotations. A single function may register
    /// under multiple procedure names (aliases). The builder synthesizes one
    /// `Procedure` node per name and emits a `Procedure -[IMPLEMENTED_BY]->
    /// Function` edge for each, letting graph-of-procedures use cases
    /// (Cypher CALL registry, RPC method catalogs, command-bus dispatchers)
    /// sit on top of the code graph.
    pub procedure_names: Vec<String>,
    /// Flexible language-specific flags (is_pymethod, is_test, ffi_kind, py_name, ...).
    pub metadata: MetadataMap,
}

/// One declared parameter on a function/method.
///
/// Excludes implicit receivers (`self`/`cls`/`&self`/`&mut self`). The parser
/// emits `ParameterKind::Variadic` for `*args`/spread/`...rest`,
/// `ParameterKind::KwVariadic` for Python `**kwargs`, and `ParameterKind::Positional`
/// for everything else. `default` carries the raw default-expression text when present.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default: Option<String>,
    pub kind: ParameterKind,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    #[default]
    Positional,
    Variadic,
    KwVariadic,
    /// Method receiver — Go's `(c *Call)`, Rust's `&self`/`&mut self`/`self`.
    /// Stored in `parameters` so the USES_TYPE resolver can attribute the
    /// receiver type with `position="receiver"`. Excluded from `param_count`
    /// since receivers aren't user-supplied arguments.
    Receiver,
}

/// Represents structs (Rust/C++), classes (Python/TS/Java/C#).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassInfo {
    pub name: String,
    pub qualified_name: String,
    /// "struct" | "class" — determines graph node type.
    pub kind: String,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub bases: Vec<String>,
    pub type_parameters: Option<String>,
    pub end_line: Option<u32>,
    pub metadata: MetadataMap,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnumInfo {
    pub name: String,
    pub qualified_name: String,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub variants: Vec<String>,
    pub end_line: Option<u32>,
    /// Structured variant info per language (C++ enum values, Rust tuple/struct variants).
    pub variant_details: Option<Vec<serde_json::Value>>,
}

/// Represents traits (Rust), protocols (Python), interfaces (TS/Java/C#).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterfaceInfo {
    pub name: String,
    pub qualified_name: String,
    /// "trait" | "protocol" | "interface"
    pub kind: String,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub type_parameters: Option<String>,
    pub end_line: Option<u32>,
}

/// Class/struct field or property.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributeInfo {
    pub name: String,
    /// Fully-qualified name, e.g. `"chromadb.Collection.name"`.
    pub qualified_name: String,
    /// Owner's qualified name, e.g. `"chromadb.Collection"`.
    pub owner_qualified_name: String,
    pub type_annotation: Option<String>,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub default_value: Option<String>,
}

/// Top-level constant, type alias, or static variable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstantInfo {
    pub name: String,
    pub qualified_name: String,
    /// "constant" | "type_alias" | "static"
    pub kind: String,
    pub type_annotation: Option<String>,
    /// First ~100 chars of the value expression.
    pub value_preview: Option<String>,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
}

/// A semantically-interesting HTML / template element — heading
/// (`h1`–`h6`), element with an `id` attribute, or a `<form>` with an
/// `action` attribute. The HTML parser (0.9.36) emits one per element
/// that fits one of those three shapes; everything else stays as parse
/// noise to keep god-HTML-file graphs navigable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElementInfo {
    /// Short display name. For headings, the text content (truncated to
    /// 100 chars). For sections, the `id` attribute value. For forms,
    /// the `action` string (or `"form_<n>"` when action is absent).
    pub name: String,
    /// Stable qualified name — `"{rel_path}:{tag}:{anchor}"`.
    pub qualified_name: String,
    /// HTML tag name (`h1`/`h2`/.../`section`/`main`/`header`/`form`/...).
    pub tag: String,
    /// `"heading"` | `"section"` | `"form"`.
    pub kind: String,
    /// `id` attribute value, when present.
    pub id: Option<String>,
    /// Form `action` attribute.
    pub action: Option<String>,
    /// Form `method` attribute (defaults to `"GET"` when absent in HTML
    /// but we record `None` to preserve "wasn't declared" vs
    /// "declared empty").
    pub method: Option<String>,
    pub file_path: String,
    pub line_number: u32,
    pub end_line: Option<u32>,
    /// Drives `Element -[CONTAINS]-> Element` edges. The qualified_name
    /// of the nearest emitted ancestor (heading/section/form). `None`
    /// for top-level elements directly under `<body>`.
    pub parent_qname: Option<String>,
}

/// A CSS rule_set selector. The 0.9.36 CSS parser emits one per
/// `rule_set`, regardless of the rule's selector-list count (a rule
/// `.foo, .bar, .baz { ... }` emits one node named `.foo, .bar, .baz`,
/// not three). Custom-property declarations are emitted separately as
/// `ConstantInfo` rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectorInfo {
    /// Canonicalised selector-list string (e.g. `".foo, .bar"`).
    pub name: String,
    /// Stable qualified name — `"{rel_path}:{line}:{selector}"`.
    pub qualified_name: String,
    /// Always `"rule"` in 0.9.36; reserved for future
    /// `"keyframes"`/`"media"`/etc. extensions.
    pub kind: String,
    pub file_path: String,
    pub line_number: u32,
    pub end_line: Option<u32>,
}

/// An impl block (Rust), inheritance (Python/Java/C#), or implements (TS).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeRelationship {
    pub source_type: String,
    /// `None` for inherent impls.
    pub target_type: Option<String>,
    /// "implements" | "extends" | "inherent"
    pub relationship: String,
    pub methods: Vec<FunctionInfo>,
}

/// Unified result of parsing one or more files.
#[derive(Debug, Clone, Default)]
pub struct ParseResult {
    pub files: Vec<FileInfo>,
    pub functions: Vec<FunctionInfo>,
    pub classes: Vec<ClassInfo>,
    pub enums: Vec<EnumInfo>,
    pub interfaces: Vec<InterfaceInfo>,
    pub type_relationships: Vec<TypeRelationship>,
    pub attributes: Vec<AttributeInfo>,
    pub constants: Vec<ConstantInfo>,
    pub elements: Vec<ElementInfo>,
    pub selectors: Vec<SelectorInfo>,
}

impl ParseResult {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the contents of `other` into `self`.
    pub fn merge(&mut self, mut other: ParseResult) {
        self.files.append(&mut other.files);
        self.functions.append(&mut other.functions);
        self.classes.append(&mut other.classes);
        self.enums.append(&mut other.enums);
        self.interfaces.append(&mut other.interfaces);
        self.type_relationships
            .append(&mut other.type_relationships);
        self.attributes.append(&mut other.attributes);
        self.constants.append(&mut other.constants);
        self.elements.append(&mut other.elements);
        self.selectors.append(&mut other.selectors);
    }
}

// ── Manifest / project-level models ────────────────────────────────

/// A directory containing source code to parse.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRoot {
    /// Absolute path to the directory.
    pub path: PathBuf,
    /// `None` = auto-detect language.
    pub language: Option<String>,
    pub is_test: bool,
    /// e.g. `"python-package"`, `"rust-crate"`.
    pub label: Option<String>,
}

/// A project dependency declared in a manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub version_spec: Option<String>,
    pub is_dev: bool,
    pub is_optional: bool,
    /// Optional dep group name.
    pub group: Option<String>,
}

/// Project metadata extracted from a manifest file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub languages: Vec<String>,
    pub authors: Vec<String>,
    pub license: Option<String>,
    pub repository_url: Option<String>,
    pub manifest_path: String,
    pub source_roots: Vec<SourceRoot>,
    pub test_roots: Vec<SourceRoot>,
    pub dependencies: Vec<DependencyInfo>,
    /// e.g. `"maturin"`, `"setuptools"`, `"cargo"`.
    pub build_system: Option<String>,
    pub metadata: MetadataMap,
}

// ── Field-embedding JSON (for ClassInfo.fields on graph nodes) ────────
//
// `fields` is serialized as a JSON string attached to Class/Struct nodes.
// Preserve exact key order: name, type, visibility, default.
// (This is verified in tests/test_code_tree_parity.py via a direct JSON
// equality assertion on a known Class node.)

/// One entry in the JSON `fields` blob on a Class/Struct node.
#[derive(Debug, Clone, Serialize)]
pub struct FieldEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub visibility: String,
    pub default: Option<String>,
}
