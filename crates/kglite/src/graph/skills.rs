//! Graph-carried skills — markdown methodology a graph carries about itself.
//!
//! Re-exported as [`kglite::api::skills`](crate::api::skills). A skill is a
//! node of the system label [`SKILL_LABEL`], so a `.kgl` file ships the
//! instructions for using it alongside the data, and an MCP server reads them
//! at boot instead of the operator wiring a skills directory.
//!
//! **Reads go straight to the node store; writes go through Cypher.** The
//! guards that make a write safe — write scope, the schema lock, declared
//! shapes, constraint checks, WAL and CDC — live on the Cypher write path and
//! nowhere else, so a raw-node upsert here would silently bypass all of them.
//! The one guard Cypher does *not* apply in core is `read_only`, which is
//! per-binding policy today; [`set`] and [`delete`] refuse on it themselves so
//! a Rust embedder gets the same answer the wheel gives.
//!
//! **All five properties are written on every upsert.** The planner's
//! typo-guard rejects a property the type's metadata has never seen, so a node
//! first written with a subset would make the next full write illegal.
//!
//! **Markdown is the interchange format**, in the frontmatter dialect
//! mcp-methods parses: [`render_markdown`] emits it and [`parse_markdown`]
//! reads it back through [`crate::okf::frontmatter`] — we never hand-roll a
//! YAML parser, which is why the parsing half only exists with the `okf`
//! feature on.

// Every entry point below reports through `KgError`, which carries structured
// query context and so trips `result_large_err` uniformly. Boxing it would
// diverge from every other core api surface a binding calls, so the allowance
// is module-scoped once rather than repeated on each function.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::path::Path;

use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy;
use crate::graph::schema::DirGraph;
use crate::graph::session::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

/// The node label every skill carries. One of
/// [`crate::graph::schema::SYSTEM_LABELS`], so skills stay out of every type
/// enumeration while remaining ordinary nodes to Cypher.
pub const SKILL_LABEL: &str = "KgliteSkill";

/// Per-skill body ceiling, matching the limit mcp-methods applies to a skill
/// loaded from a file. A body past it is refused at [`validate`] rather than
/// at the far end of a save, a load and a server boot.
pub const MAX_BODY_BYTES: usize = 16 * 1024;

/// When the agent host is handed the skill's body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Delivery {
    /// Body inlined in the tool description at boot.
    Eager,
    /// Only the name and description are advertised; the body is fetched when
    /// the agent asks for it.
    #[default]
    Lazy,
}

impl Delivery {
    /// The stored spelling — this is what lands on the node and in frontmatter.
    pub fn as_str(self) -> &'static str {
        match self {
            Delivery::Eager => "eager",
            Delivery::Lazy => "lazy",
        }
    }

    /// Parse a stored spelling. Unknown values are refused rather than
    /// defaulted: a typo would otherwise silently demote an eager skill.
    pub fn parse(text: &str) -> Result<Self, KgError> {
        match text {
            "eager" => Ok(Delivery::Eager),
            "lazy" => Ok(Delivery::Lazy),
            other => Err(KgError::InvalidArgument {
                argument: "delivery".to_string(),
                expected: "'eager' or 'lazy'".to_string(),
                found: other.to_string(),
            }),
        }
    }
}

/// One skill. `name` is the key: at most one node per name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkillRecord {
    pub name: String,
    pub description: String,
    pub body: String,
    pub references_tools: Vec<String>,
    pub delivery: Delivery,
}

/// What [`set`] did — the caller usually wants to report one or the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetOutcome {
    Created,
    Updated,
}

/// Check a record against the rules a hand-written `CREATE` cannot enforce.
///
/// `name` is a filename stem on export and a registry key at boot, so it must
/// be a single path-safe token: no separator, no whitespace, no `.` traversal.
pub fn validate(record: &SkillRecord) -> Result<(), KgError> {
    if record.name.trim().is_empty() {
        return Err(KgError::InvalidArgument {
            argument: "name".to_string(),
            expected: "a non-empty skill name".to_string(),
            found: "empty".to_string(),
        });
    }
    let unsafe_name = record
        .name
        .chars()
        .any(|c| c.is_whitespace() || c == '/' || c == '\\' || c == ':' || c.is_control())
        || record.name == "."
        || record.name == "..";
    if unsafe_name {
        return Err(KgError::InvalidArgument {
            argument: "name".to_string(),
            expected: "a single path-safe token (no whitespace, '/', '\\' or ':')".to_string(),
            found: record.name.clone(),
        });
    }
    if record.description.trim().is_empty() {
        return Err(KgError::InvalidArgument {
            argument: "description".to_string(),
            expected:
                "a non-empty description — it is all an agent sees before asking for the body"
                    .to_string(),
            found: "empty".to_string(),
        });
    }
    if record.body.len() > MAX_BODY_BYTES {
        return Err(KgError::InvalidArgument {
            argument: "body".to_string(),
            expected: format!("at most {MAX_BODY_BYTES} bytes"),
            found: format!("{} bytes", record.body.len()),
        });
    }
    Ok(())
}

// ── Reads ──────────────────────────────────────────────────────────────────

fn string_property(graph: &DirGraph, idx: petgraph::graph::NodeIndex, key: &str) -> String {
    let Some(view) = graph.graph.node_view(idx) else {
        return String::new();
    };
    match view.get_property_value(key) {
        Some(Value::String(s)) => s,
        // A skill written by any other route may have stored a non-string; the
        // display form is still better than dropping the skill entirely.
        Some(Value::Null) | None => String::new(),
        Some(other) => crate::datatypes::values::raw_string(&other),
    }
}

fn tools_property(graph: &DirGraph, idx: petgraph::graph::NodeIndex) -> Vec<String> {
    let Some(view) = graph.graph.node_view(idx) else {
        return Vec::new();
    };
    match view.get_property_value("references_tools") {
        Some(Value::List(items)) => items
            .iter()
            .map(crate::datatypes::values::raw_string)
            .collect(),
        Some(Value::String(one)) => vec![one],
        _ => Vec::new(),
    }
}

fn read_record(graph: &DirGraph, idx: petgraph::graph::NodeIndex, with_body: bool) -> SkillRecord {
    SkillRecord {
        name: string_property(graph, idx, "name"),
        description: string_property(graph, idx, "description"),
        body: if with_body {
            string_property(graph, idx, "body")
        } else {
            String::new()
        },
        references_tools: tools_property(graph, idx),
        delivery: Delivery::parse(&string_property(graph, idx, "delivery")).unwrap_or_default(),
    }
}

/// Every skill in the graph, sorted by name, **without bodies** — the catalogue
/// an agent host advertises. A body can run to 16 KiB, so a listing that
/// carried them would hand a caller megabytes it did not ask for; use [`get`]
/// for the one that matters.
pub fn list(graph: &DirGraph) -> Vec<SkillRecord> {
    let Some(members) = graph.type_indices.get(SKILL_LABEL) else {
        return Vec::new();
    };
    let mut out: Vec<SkillRecord> = members
        .iter()
        .map(|idx| read_record(graph, idx, false))
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// One skill, body included.
pub fn get(graph: &DirGraph, name: &str) -> Result<SkillRecord, KgError> {
    let found = graph.type_indices.get(SKILL_LABEL).and_then(|members| {
        members
            .iter()
            .find(|idx| string_property(graph, *idx, "name") == name)
    });
    match found {
        Some(idx) => Ok(read_record(graph, idx, true)),
        None => Err(KgError::NodeNotFound {
            node_type: SKILL_LABEL.to_string(),
            id: name.to_string(),
        }),
    }
}

// ── Writes ─────────────────────────────────────────────────────────────────

fn skill_opts(params: &HashMap<String, Value>) -> ExecuteOptions<'_> {
    ExecuteOptions {
        params,
        deadline: None,
        deadline_origin: None,
        max_work_units: None,
        row_limit: None,
        lazy_eligible: false,
        streaming: true,
        parallel: false,
        hide_structural_keys: false,
        disabled_passes: None,
        embedder: None,
        value_codecs: None,
        cancel: None,
        write_scope: None,
        git_sha: None,
        modified_by: None,
        csv_import: CsvImportPolicy::Denied,
    }
}

fn refuse_if_read_only(graph: &DirGraph) -> Result<(), KgError> {
    if graph.read_only {
        return Err(KgError::read_only(
            "Graph is in read-only mode — skills cannot be created, updated or \
             deleted. Re-enable mutations before writing skills.",
        ));
    }
    Ok(())
}

/// Create or replace the skill named `record.name`.
///
/// Routes through Cypher `MERGE` so the write inherits the schema lock, write
/// scope, declared shapes, constraint checks, WAL and CDC. All five properties
/// are written every time — see the module docs for why a partial write is not
/// an option.
pub fn set(graph: &mut DirGraph, record: &SkillRecord) -> Result<SetOutcome, KgError> {
    validate(record)?;
    refuse_if_read_only(graph)?;

    let existed = get(graph, &record.name).is_ok();

    let props: Vec<(crate::datatypes::PropKey, Value)> = vec![
        ("body".into(), Value::String(record.body.clone())),
        (
            "delivery".into(),
            Value::String(record.delivery.as_str().to_string()),
        ),
        (
            "description".into(),
            Value::String(record.description.clone()),
        ),
        (
            "references_tools".into(),
            Value::List(
                record
                    .references_tools
                    .iter()
                    .map(|t| Value::String(t.clone()))
                    .collect(),
            ),
        ),
    ];

    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("name".to_string(), Value::String(record.name.clone()));
    params.insert(
        "props".to_string(),
        Value::Map(crate::datatypes::PropMap::from_pairs(props)),
    );

    execute_mut(
        graph,
        &format!("MERGE (s:{SKILL_LABEL} {{name: $name}}) SET s += $props"),
        &skill_opts(&params),
    )?;

    Ok(if existed {
        SetOutcome::Updated
    } else {
        SetOutcome::Created
    })
}

/// Remove the skill named `name`. `false` means there was nothing to remove.
pub fn delete(graph: &mut DirGraph, name: &str) -> Result<bool, KgError> {
    refuse_if_read_only(graph)?;
    if get(graph, name).is_err() {
        return Ok(false);
    }
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("name".to_string(), Value::String(name.to_string()));
    execute_mut(
        graph,
        &format!("MATCH (s:{SKILL_LABEL} {{name: $name}}) DETACH DELETE s"),
        &skill_opts(&params),
    )?;
    Ok(true)
}

// ── Markdown ───────────────────────────────────────────────────────────────

/// A YAML double-quoted scalar. JSON's string grammar is a subset of YAML's
/// double-quoted one, so the encoder we already depend on emits a scalar every
/// YAML parser accepts — and emitting is the only half of YAML this module
/// does itself.
fn yaml_quoted(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| format!("\"{}\"", text.replace('"', "'")))
}

/// Render a skill as a SKILL.md document: frontmatter in the dialect
/// mcp-methods parses, then the body after the closing `---`.
pub fn render_markdown(record: &SkillRecord) -> String {
    let mut out = String::with_capacity(record.body.len() + 256);
    out.push_str("---\n");
    out.push_str(&format!("name: {}\n", yaml_quoted(&record.name)));
    out.push_str(&format!(
        "description: {}\n",
        yaml_quoted(&record.description)
    ));
    out.push_str("references_tools:\n");
    for tool in &record.references_tools {
        out.push_str(&format!("  - {}\n", yaml_quoted(tool)));
    }
    out.push_str(&format!(
        "delivery: {}\n",
        yaml_quoted(record.delivery.as_str())
    ));
    out.push_str("---\n\n");
    out.push_str(&record.body);
    out
}

/// Read a SKILL.md document. `name` falls back to nothing — a file whose
/// frontmatter omits it fails [`validate`] with the missing field named.
///
/// An empty `references_tools:` key parses as a null scalar rather than an
/// empty sequence, so a non-list value is read as "no tools" instead of being
/// refused.
#[cfg(feature = "okf")]
pub fn parse_markdown(text: &str) -> Result<SkillRecord, KgError> {
    let (_, body) = crate::okf::frontmatter::split(text);
    let front = crate::okf::frontmatter::parse(text).map_err(KgError::Argument)?;

    let scalar = |key: &str| -> String {
        match front.get(key) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => crate::datatypes::values::raw_string(other),
        }
    };
    let delivery_text = scalar("delivery");
    let record = SkillRecord {
        name: scalar("name"),
        description: scalar("description"),
        body: body.trim_start_matches('\n').to_string(),
        references_tools: match front.get("references_tools") {
            Some(Value::List(items)) => items
                .iter()
                .map(crate::datatypes::values::raw_string)
                .collect(),
            Some(Value::String(one)) => vec![one.clone()],
            _ => Vec::new(),
        },
        delivery: if delivery_text.is_empty() {
            Delivery::default()
        } else {
            Delivery::parse(&delivery_text)?
        },
    };
    validate(&record)?;
    Ok(record)
}

/// Upsert every skill under `path` — one `.md` file, or a directory of them.
///
/// Directory reads are non-recursive and sorted, so the upsert order is the
/// same on every platform. Returns the names imported, in that order.
#[cfg(feature = "okf")]
pub fn import_path(graph: &mut DirGraph, path: &Path) -> Result<Vec<String>, KgError> {
    let meta = std::fs::metadata(path).map_err(|_| KgError::FileNotFound(path.to_path_buf()))?;
    let mut files: Vec<std::path::PathBuf> = if meta.is_dir() {
        let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .map_err(KgError::FileIo)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "md"))
            .collect();
        found.sort();
        found
    } else {
        vec![path.to_path_buf()]
    };
    files.dedup();

    let mut names = Vec::with_capacity(files.len());
    for file in files {
        let text = std::fs::read_to_string(&file).map_err(KgError::FileIo)?;
        let record = parse_markdown(&text).map_err(|err| KgError::FileFormat {
            path: file.clone(),
            message: err.to_string(),
        })?;
        set(graph, &record)?;
        names.push(record.name);
    }
    Ok(names)
}

/// Write every skill to `dir` as `<name>.md`, creating the directory if it does
/// not exist. Returns the names written, sorted.
pub fn export_dir(graph: &DirGraph, dir: &Path) -> Result<Vec<String>, KgError> {
    std::fs::create_dir_all(dir).map_err(KgError::FileIo)?;
    let mut names = Vec::new();
    for summary in list(graph) {
        let record = get(graph, &summary.name)?;
        std::fs::write(
            dir.join(format!("{}.md", record.name)),
            render_markdown(&record),
        )
        .map_err(KgError::FileIo)?;
        names.push(record.name);
    }
    Ok(names)
}

#[cfg(test)]
#[path = "skills_tests.rs"]
mod tests;
