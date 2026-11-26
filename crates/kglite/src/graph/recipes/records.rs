//! Graph-carried recipe records — one catalogue query per node.
//!
//! A `.kgl` file that carries its own recipes ships the exact queries its
//! skills name, so an agent host serves them without the operator copying
//! Cypher into a manifest. The node shape is the catalogue shape taken apart:
//! `recipe` + `name` are the key, `recipe_description` is the group's text
//! repeated on every member, and the remaining three fields are the query's.
//!
//! **Reads go straight to the node store; writes go through Cypher.** Same
//! reasoning as [`crate::graph::skills`]: the schema lock, write scope,
//! declared shapes, constraint checks, WAL and CDC all live on the Cypher
//! write path, and `read_only` is refused here because core's `execute_mut`
//! does not check it.
//!
//! **A record is held to the rules the catalogue applies**, not a looser
//! subset: [`validate`] compiles the query exactly as `from_manifest_value`
//! would, so a record that would be skipped at boot is refused where it is
//! written.

// Every entry point below reports through `KgError`, which carries structured
// query context and so trips `result_large_err` uniformly. Boxing it would
// diverge from `graph::skills` and from every other core api surface a binding
// calls, so the allowance is module-scoped once rather than per function.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::path::Path;

use serde_json::{Map, Value as Json};

use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy;
use crate::graph::schema::DirGraph;
use crate::graph::session::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

use super::{validate_identifier, RecipeCatalog, RecipeCatalogError, RecipeQueryDefinition};

/// The node label every graph-carried recipe query carries. One of
/// [`crate::graph::schema::SYSTEM_LABELS`], so recipes stay out of every type
/// enumeration while remaining ordinary nodes to Cypher.
pub const RECIPE_LABEL: &str = "KgliteRecipe";

/// One stored catalogue query, as it lives on a node.
///
/// `parameters` is the query's JSON Schema, stored as a **native nested map**
/// rather than as an encoded JSON string, so `r.parameters.type` is readable
/// from Cypher like any other property. That choice is measured, not assumed:
/// `parameters_survive_every_storage_mode_and_a_kgl_round_trip` writes a schema
/// with nested `properties`, `items`, a mixed-type `enum`, an integer `minimum`
/// beside a float `maximum` and a non-alphabetical `required`, in Memory,
/// Mapped and Disk modes, through a `.kgl` save and load, and asserts
/// `serde_json::Value` equality on the way back. Nothing is lost — including
/// the `1` versus `1.0` distinction the schema compiler's numeric-bound rule
/// depends on, and the empty `properties: {}` / `required: []` of a
/// parameter-free query.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecipeRecord {
    /// Group id. A catalogue identifier: `^[A-Za-z_][A-Za-z0-9_]*$`.
    pub recipe: String,
    /// Query id within the group, with the same identifier rule.
    pub name: String,
    /// What this query answers — what an agent reads before calling it.
    pub description: String,
    /// The query's JSON Schema, as a `serde_json` document.
    pub parameters: Json,
    /// The stored, parameterised, read-only Cypher.
    pub cypher: String,
    /// The group's description. Every member of a group carries it; the
    /// catalogue takes the first in name order.
    pub recipe_description: String,
    /// The MCP tool name this query is served under, when its author asked
    /// for one. `None` — the default — leaves it reachable through
    /// `run_recipe_query` alone.
    pub tool: Option<String>,
}

/// What [`set`] did — the caller usually wants to report one or the other.
///
/// Deliberately its own type rather than a shared upsert enum: a recipe and a
/// skill are independent api surfaces, and nothing reads one outcome as the
/// other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetOutcome {
    Created,
    Updated,
}

/// Why one graph-carried record could not enter the catalogue.
///
/// [`catalogue_from_graph`] returns these instead of failing: a manifest typo
/// is the operator's to fix at boot, but graph content is data, and one bad
/// node must not take the rest of the catalogue down with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeWarning {
    pub recipe: String,
    pub name: String,
    pub reason: String,
}

impl std::fmt::Display for RecipeWarning {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "recipe {:?} query {:?} skipped: {}",
            self.recipe, self.name, self.reason
        )
    }
}

fn missing(argument: &str, expected: &str) -> KgError {
    KgError::InvalidArgument {
        argument: argument.to_string(),
        expected: expected.to_string(),
        found: "empty".to_string(),
    }
}

/// Check a record against the rules a hand-written `CREATE` cannot enforce:
/// the two identifiers, the three non-empty texts, and then the full catalogue
/// compile — read-only Cypher, the closed schema-keyword set, and `$params`
/// matching `properties`/`required` exactly.
pub fn validate(record: &RecipeRecord) -> Result<(), KgError> {
    validate_identifier(&record.recipe, "recipe")?;
    validate_identifier(&record.name, "query")?;
    if record.description.trim().is_empty() {
        return Err(missing(
            "description",
            "a non-empty description — it is what an agent reads before calling the query",
        ));
    }
    if record.recipe_description.trim().is_empty() {
        return Err(missing(
            "recipe_description",
            "a non-empty group description — the catalogue requires one per recipe",
        ));
    }
    if record.cypher.trim().is_empty() {
        return Err(missing("cypher", "a non-empty Cypher statement"));
    }
    if let Some(tool) = &record.tool {
        super::validate_tool_name(tool)?;
    }
    compile(record)?;
    Ok(())
}

fn compile(record: &RecipeRecord) -> Result<RecipeQueryDefinition, RecipeCatalogError> {
    RecipeQueryDefinition::compile(
        &record.name,
        record.description.clone(),
        record.cypher.clone(),
        &record.parameters,
        record.tool.clone(),
    )
}

// ── Reads ──────────────────────────────────────────────────────────────────

fn string_property(graph: &DirGraph, idx: petgraph::graph::NodeIndex, key: &str) -> String {
    let Some(view) = graph.graph.node_view(idx) else {
        return String::new();
    };
    match view.get_property_value(key) {
        Some(Value::String(s)) => s,
        // A record written by any other route may have stored a non-string;
        // the raw form still reaches validation, which names what is wrong.
        Some(Value::Null) | None => String::new(),
        Some(other) => crate::datatypes::values::raw_string(&other),
    }
}

/// Read `parameters` back into a `serde_json` document.
///
/// A node whose `parameters` is not a map at all — a hand-written `CREATE`
/// that stored a string, say — converts to whatever it is and fails the schema
/// compile with the record named, rather than disappearing.
fn parameters_property(graph: &DirGraph, idx: petgraph::graph::NodeIndex) -> Json {
    let Some(view) = graph.graph.node_view(idx) else {
        return Json::Null;
    };
    match view.get_property_value("parameters") {
        Some(value) => crate::param::kglite_value_to_json(&value),
        None => Json::Null,
    }
}

fn read_record(graph: &DirGraph, idx: petgraph::graph::NodeIndex) -> RecipeRecord {
    RecipeRecord {
        recipe: string_property(graph, idx, "recipe"),
        name: string_property(graph, idx, "name"),
        description: string_property(graph, idx, "description"),
        parameters: parameters_property(graph, idx),
        cypher: string_property(graph, idx, "cypher"),
        recipe_description: string_property(graph, idx, "recipe_description"),
        tool: Some(string_property(graph, idx, "tool")).filter(|name| !name.is_empty()),
    }
}

/// Every recipe record in the graph, sorted by `(recipe, name)`, **complete**.
///
/// Unlike [`crate::graph::skills::list`], which drops 16 KiB bodies, this keeps
/// every field: a recipe query is one statement, and both the catalogue build
/// and export need every property, so an abridged listing would only force a
/// [`get`] per row.
pub fn list(graph: &DirGraph) -> Vec<RecipeRecord> {
    let Some(members) = graph.type_indices.get(RECIPE_LABEL) else {
        return Vec::new();
    };
    let mut out: Vec<RecipeRecord> = members.iter().map(|idx| read_record(graph, idx)).collect();
    out.sort_by(|a, b| a.recipe.cmp(&b.recipe).then_with(|| a.name.cmp(&b.name)));
    out
}

/// One record by its `(recipe, name)` key.
pub fn get(graph: &DirGraph, recipe: &str, name: &str) -> Result<RecipeRecord, KgError> {
    let found = graph.type_indices.get(RECIPE_LABEL).and_then(|members| {
        members.iter().find(|idx| {
            string_property(graph, *idx, "recipe") == recipe
                && string_property(graph, *idx, "name") == name
        })
    });
    match found {
        Some(idx) => Ok(read_record(graph, idx)),
        None => Err(KgError::NodeNotFound {
            node_type: RECIPE_LABEL.to_string(),
            id: format!("{recipe}/{name}"),
        }),
    }
}

// ── Writes ─────────────────────────────────────────────────────────────────

fn recipe_opts(params: &HashMap<String, Value>) -> ExecuteOptions<'_> {
    ExecuteOptions {
        hide_structural_keys: false,
        params,
        deadline: None,
        deadline_origin: None,
        max_work_units: None,
        row_limit: None,
        lazy_eligible: false,
        streaming: true,
        parallel: false,
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
            "Graph is in read-only mode — recipes cannot be created, updated or \
             deleted. Re-enable mutations before writing recipes.",
        ));
    }
    Ok(())
}

/// Create or replace the record keyed `(record.recipe, record.name)`.
///
/// Routes through Cypher `MERGE` so the write inherits the schema lock, write
/// scope, declared shapes, constraint checks, WAL and CDC. Every property is
/// written every time — including `tool` as an empty string when the record
/// declares none: the planner's typo-guard rejects a property the type's
/// metadata has never seen, so a node first written with a subset would make
/// the next full write illegal, and an omitted `tool` would leave a cleared
/// one still serving.
pub fn set(graph: &mut DirGraph, record: &RecipeRecord) -> Result<SetOutcome, KgError> {
    validate(record)?;
    refuse_if_read_only(graph)?;

    let existed = get(graph, &record.recipe, &record.name).is_ok();

    let props: Vec<(crate::datatypes::PropKey, Value)> = vec![
        ("cypher".into(), Value::String(record.cypher.clone())),
        (
            "description".into(),
            Value::String(record.description.clone()),
        ),
        ("name".into(), Value::String(record.name.clone())),
        (
            "parameters".into(),
            crate::param::json_value_to_kglite_value(&record.parameters),
        ),
        ("recipe".into(), Value::String(record.recipe.clone())),
        (
            "recipe_description".into(),
            Value::String(record.recipe_description.clone()),
        ),
        (
            "tool".into(),
            Value::String(record.tool.clone().unwrap_or_default()),
        ),
    ];

    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("recipe".to_string(), Value::String(record.recipe.clone()));
    params.insert("name".to_string(), Value::String(record.name.clone()));
    params.insert(
        "props".to_string(),
        Value::Map(crate::datatypes::PropMap::from_pairs(props)),
    );

    execute_mut(
        graph,
        &format!("MERGE (r:{RECIPE_LABEL} {{recipe: $recipe, name: $name}}) SET r += $props"),
        &recipe_opts(&params),
    )?;

    Ok(if existed {
        SetOutcome::Updated
    } else {
        SetOutcome::Created
    })
}

/// Remove the record keyed `(recipe, name)`. `false` means there was nothing
/// to remove.
pub fn delete(graph: &mut DirGraph, recipe: &str, name: &str) -> Result<bool, KgError> {
    refuse_if_read_only(graph)?;
    if get(graph, recipe, name).is_err() {
        return Ok(false);
    }
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("recipe".to_string(), Value::String(recipe.to_string()));
    params.insert("name".to_string(), Value::String(name.to_string()));
    execute_mut(
        graph,
        &format!("MATCH (r:{RECIPE_LABEL} {{recipe: $recipe, name: $name}}) DETACH DELETE r"),
        &recipe_opts(&params),
    )?;
    Ok(true)
}

// ── Catalogue ──────────────────────────────────────────────────────────────

/// Compile every valid record into a catalogue, reporting the rest.
///
/// A group's description is the first member's in name order — [`list`] is
/// sorted, and [`RecipeCatalog::insert_query`] keeps the description it was
/// first given.
pub fn catalogue_from_graph(graph: &DirGraph) -> (RecipeCatalog, Vec<RecipeWarning>) {
    let mut catalogue = RecipeCatalog::default();
    let mut warnings = Vec::new();
    for record in list(graph) {
        let compiled = validate(&record).and_then(|()| compile(&record).map_err(KgError::from));
        match compiled {
            Ok(query) => catalogue.insert_query(&record.recipe, &record.recipe_description, query),
            Err(error) => warnings.push(RecipeWarning {
                recipe: record.recipe,
                name: record.name,
                reason: error.to_string(),
            }),
        }
    }
    (catalogue, warnings)
}

// ── Import / export ────────────────────────────────────────────────────────

/// The `extensions.cypher_recipes` document for everything in the graph.
///
/// Every record is exported, including one that no longer compiles, so an
/// operator can export, repair and re-import rather than losing the text.
pub fn export_value(graph: &DirGraph) -> Json {
    let mut recipes: Map<String, Json> = Map::new();
    for record in list(graph) {
        let entry = recipes.entry(record.recipe.clone()).or_insert_with(|| {
            serde_json::json!({
                "description": record.recipe_description.clone(),
                "queries": Json::Object(Map::new()),
            })
        });
        let Some(queries) = entry.get_mut("queries").and_then(Json::as_object_mut) else {
            continue;
        };
        let mut query = serde_json::json!({
            "description": record.description,
            "parameters": record.parameters,
            "cypher": record.cypher,
        });
        if let (Some(tool), Some(map)) = (record.tool, query.as_object_mut()) {
            map.insert("tool".to_string(), Json::String(tool));
        }
        queries.insert(record.name.clone(), query);
    }
    Json::Object(recipes)
}

/// Read one recipe query from the markdown dialect a vault carries
/// (VAULT.md §8): frontmatter `recipe`, `name`, `description`, optional
/// `recipe_description`, `parameters` and `tool`, with the statement in the
/// body's single ` ```cypher ` fence.
///
/// **Not validated here.** `recipe_description` is optional in the file and
/// required by [`validate`], and the missing one is inherited from the group
/// the record joins — which needs a graph. [`set_from_markdown`] does that and
/// then validates; a caller using this directly owns the same step.
///
/// `parameters` is read as a **nested** value
/// ([`crate::okf::frontmatter::parse_yaml`]), not through the flattening
/// frontmatter reader: a JSON Schema's `properties.id.type` is three levels,
/// and a dotted key would make it one property named `properties.id.type`.
#[cfg(feature = "okf")]
pub fn parse_markdown(text: &str) -> Result<RecipeRecord, KgError> {
    let (yaml, body) = crate::okf::frontmatter::split(text);
    let front = crate::okf::frontmatter::parse_yaml(yaml.as_deref().unwrap_or_default())
        .map_err(KgError::Argument)?;
    let front = match &front {
        Value::Map(map) => map.clone(),
        _ => {
            return Err(KgError::Argument(
                "a recipe file needs a YAML frontmatter block naming `recipe`, `name` and \
                 `description`"
                    .to_string(),
            ))
        }
    };
    let scalar = |key: &str| -> String {
        match front.get(key) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => crate::datatypes::values::raw_string(other),
        }
    };
    Ok(RecipeRecord {
        recipe: scalar("recipe"),
        name: scalar("name"),
        description: scalar("description"),
        recipe_description: scalar("recipe_description"),
        parameters: match front.get("parameters") {
            Some(Value::Null) | None => empty_schema(),
            Some(value) => crate::param::kglite_value_to_json(value),
        },
        cypher: cypher_fence(&body)?,
        tool: Some(scalar("tool")).filter(|name| !name.is_empty()),
    })
}

/// Render a recipe query as the `.kglite/recipes/*.md` document
/// [`parse_markdown`] reads (VAULT.md §8).
///
/// `parameters` is emitted as JSON, which is YAML flow syntax: the schema is a
/// nested document, and a flow mapping keeps it on one line without this
/// module having to own a YAML block emitter as well.
#[cfg(feature = "okf")]
pub fn render_markdown(record: &RecipeRecord) -> String {
    let quoted = |text: &str| -> String {
        serde_json::to_string(text).unwrap_or_else(|_| format!("\"{}\"", text.replace('"', "'")))
    };
    let parameters = serde_json::to_string(&record.parameters).unwrap_or_else(|_| "{}".to_string());
    // Emitted only when set: an author reading the file back must see the key
    // exactly when the query is served as a tool.
    let tool = record
        .tool
        .as_deref()
        .map(|name| format!("tool: {}\n", quoted(name)))
        .unwrap_or_default();
    format!(
        "---\nrecipe: {}\nname: {}\ndescription: {}\nrecipe_description: {}\nparameters: {}\n{tool}---\n\n```cypher\n{}\n```\n",
        quoted(&record.recipe),
        quoted(&record.name),
        quoted(&record.description),
        quoted(&record.recipe_description),
        parameters,
        record.cypher.trim(),
    )
}

/// The statement inside the body's single ` ```cypher ` fence.
///
/// Exactly one: a file with none has nothing to store, and a file with two has
/// not said which is the query — both are refused with the count, rather than
/// silently storing the first.
#[cfg(feature = "okf")]
fn cypher_fence(body: &str) -> Result<String, KgError> {
    let mut blocks: Vec<String> = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in body.lines() {
        let trimmed = line.trim();
        match &mut current {
            // A closing fence is any ``` — the language tag is opening-only.
            Some(lines) if trimmed.starts_with("```") => {
                blocks.push(lines.join("\n"));
                current = None;
            }
            Some(lines) => lines.push(line),
            None => {
                let tag = trimmed.strip_prefix("```").map(str::trim).unwrap_or("");
                if trimmed.starts_with("```") && tag.eq_ignore_ascii_case("cypher") {
                    current = Some(Vec::new());
                }
            }
        }
    }
    match blocks.len() {
        1 => Ok(blocks.remove(0).trim().to_string()),
        0 => Err(KgError::Argument(
            "a recipe file's body must hold one ```cypher fenced block; found none".to_string(),
        )),
        n => Err(KgError::Argument(format!(
            "a recipe file's body must hold exactly one ```cypher fenced block; found {n}"
        ))),
    }
}

/// The closed, parameter-free JSON Schema a file that declares no
/// `parameters:` stores — the same document the wheel's `set_recipe` writes,
/// so the two routes produce byte-identical nodes.
#[cfg(feature = "okf")]
fn empty_schema() -> Json {
    let mut schema = Map::new();
    schema.insert("type".to_string(), Json::String("object".to_string()));
    schema.insert("properties".to_string(), Json::Object(Map::new()));
    schema.insert("required".to_string(), Json::Array(Vec::new()));
    schema.insert("additionalProperties".to_string(), Json::Bool(false));
    Json::Object(schema)
}

/// Fill each record's omitted `recipe_description` from its group (VAULT.md
/// §8), and return the groups nothing described — in first-seen order, once
/// each.
///
/// **The whole batch is read before any record is filled**, so the
/// inheritance does not depend on the order the files arrived in: a group
/// whose description is written in its alphabetically last file resolves
/// exactly like one that writes it in the first. Resolving per record as it
/// was read is what skipped five of the P16 probe's six sibling files for
/// "expected a non-empty group description" — the sixth declared it and
/// sorted last.
///
/// A group already stored in `graph` is the fallback, which is what lets a
/// second import add one query to a group the graph already carries.
///
/// A record left without a description is **not** failed here: the callers
/// differ on what that costs — a directory import refuses the whole batch,
/// and a vault build skips the file with a warning (§8) — so each applies its
/// own posture to the groups this returns.
#[cfg(feature = "okf")]
pub(crate) fn inherit_group_descriptions(
    records: &mut [RecipeRecord],
    graph: &DirGraph,
) -> Vec<String> {
    let mut declared: HashMap<String, String> = HashMap::new();
    for record in records.iter() {
        if !record.recipe_description.trim().is_empty() {
            declared
                .entry(record.recipe.clone())
                .or_insert_with(|| record.recipe_description.clone());
        }
    }
    if records
        .iter()
        .any(|record| record.recipe_description.trim().is_empty())
    {
        for stored in list(graph) {
            if !stored.recipe_description.trim().is_empty() {
                declared
                    .entry(stored.recipe)
                    .or_insert(stored.recipe_description);
            }
        }
    }

    let mut undescribed: Vec<String> = Vec::new();
    for record in records.iter_mut() {
        if !record.recipe_description.trim().is_empty() {
            continue;
        }
        match declared.get(&record.recipe) {
            Some(description) => record.recipe_description = description.clone(),
            None if !undescribed.contains(&record.recipe) => {
                undescribed.push(record.recipe.clone())
            }
            None => {}
        }
    }
    undescribed
}

/// [`parse_markdown`] then [`set`], inheriting an omitted group description
/// from a query already stored under the same `recipe`.
///
/// The inheritance is the wheel's `set_recipe` rule, in core because a vault's
/// `.kglite/recipes/` is a directory of sibling files and only one of them has
/// to spell the group out. One file is its own batch, so the graph is the only
/// place [`inherit_group_descriptions`] can read it from here.
#[cfg(feature = "okf")]
pub fn set_from_markdown(graph: &mut DirGraph, text: &str) -> Result<RecipeRecord, KgError> {
    let mut records = [parse_markdown(text)?];
    inherit_group_descriptions(&mut records, graph);
    let [record] = records;
    set(graph, &record)?;
    Ok(record)
}

/// Upsert every query in an `extensions.cypher_recipes` document.
///
/// Accepts a whole manifest (the catalogue is read from `extensions`) or a bare
/// catalogue mapping. The document is compiled before anything is written, so
/// an invalid file leaves the graph untouched. Returns the `(recipe, name)`
/// keys written, in catalogue order.
pub fn import_value(
    graph: &mut DirGraph,
    document: &Json,
) -> Result<Vec<(String, String)>, KgError> {
    let raw = catalogue_section(document);
    let catalogue = RecipeCatalog::from_manifest_value(raw).map_err(KgError::from)?;

    let mut written = Vec::new();
    for recipe in catalogue.recipes() {
        for query in recipe.queries() {
            let record = RecipeRecord {
                recipe: recipe.name.clone(),
                name: query.name.clone(),
                description: query.description.clone(),
                parameters: Json::Object(query.parameters.as_json().clone()),
                cypher: query.cypher.clone(),
                recipe_description: recipe.description.clone(),
                tool: query.tool.clone(),
            };
            set(graph, &record)?;
            written.push((record.recipe, record.name));
        }
    }
    Ok(written)
}

/// A manifest nests the catalogue under `extensions.cypher_recipes`; a bare
/// catalogue file is the mapping itself. `extensions` is not a legal recipe
/// identifier, so the two shapes cannot be confused.
fn catalogue_section(document: &Json) -> Option<&Json> {
    match document.get("extensions") {
        Some(extensions) => extensions.get("cypher_recipes"),
        None => Some(document),
    }
}

/// Import recipe queries from a `.json` catalogue document, a `.md` recipe
/// file, or a directory of `.md` files (VAULT.md §8's dialect).
///
/// A **YAML catalogue** is still not read here. Core links no general YAML
/// reader for that shape: `okf`'s frontmatter helper flattens nested mappings
/// into dotted keys and renders any number outside `i64` as a float lexeme,
/// and a catalogue's parameter schemas depend on exactly those two
/// distinctions. (The markdown dialect escapes that because its schema goes
/// through [`crate::okf::frontmatter::parse_yaml`], which does not flatten.)
/// A YAML catalogue is read by whoever already has a YAML parser — the MCP
/// server reads the manifest — and handed here as a [`Json`] value.
///
/// Markdown imports are all-or-nothing like the JSON one: every file is parsed
/// before any is written, so a directory with one bad file leaves the graph
/// untouched. (A *vault* build takes the opposite posture — §8 has one bad
/// file skipped with a warning and its siblings loaded — because there the
/// directory is hand-authored content rather than a catalogue the caller
/// chose to install.)
pub fn import_path(graph: &mut DirGraph, path: &Path) -> Result<Vec<(String, String)>, KgError> {
    let meta = std::fs::metadata(path).map_err(|_| KgError::FileNotFound(path.to_path_buf()))?;
    if meta.is_dir() {
        return import_markdown_dir(graph, path);
    }
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "json" => {
            let text = std::fs::read_to_string(path)
                .map_err(|_| KgError::FileNotFound(path.to_path_buf()))?;
            let document: Json =
                serde_json::from_str(&text).map_err(|error| KgError::FileFormat {
                    path: path.to_path_buf(),
                    message: error.to_string(),
                })?;
            import_value(graph, &document)
        }
        "md" => import_markdown_dir(graph, path),
        _ => Err(KgError::FileFormat {
            path: path.to_path_buf(),
            message: "recipe import reads a .json catalogue, a .md recipe file, or a \
                      directory of them; convert a YAML catalogue first, or pass the \
                      parsed document to import_value"
                .to_string(),
        }),
    }
}

/// Import one `.md` recipe file, or every `.md` directly inside a directory.
#[cfg(feature = "okf")]
fn import_markdown_dir(
    graph: &mut DirGraph,
    path: &Path,
) -> Result<Vec<(String, String)>, KgError> {
    let files: Vec<std::path::PathBuf> = if path.is_dir() {
        let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .map_err(KgError::FileIo)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "md"))
            .collect();
        // Sorted, so the order the records are written in — and therefore
        // which file a duplicate `(recipe, name)` ends up as — does not
        // depend on directory order.
        found.sort();
        found
    } else {
        vec![path.to_path_buf()]
    };

    // Parsed and validated in full before anything is written. The group
    // descriptions the files inherit come from the batch itself, so a
    // directory installs the same way whether or not the graph already holds
    // the group.
    let mut records: Vec<RecipeRecord> = Vec::with_capacity(files.len());
    for file in &files {
        let text = std::fs::read_to_string(file).map_err(KgError::FileIo)?;
        records.push(parse_markdown(&text).map_err(|err| KgError::FileFormat {
            path: file.clone(),
            message: err.to_string(),
        })?);
    }
    // One error per undescribed *group*, not per member file: the author has
    // one `recipe_description` to write however many queries the group holds.
    if let Some(group) = inherit_group_descriptions(&mut records, graph).first() {
        return Err(KgError::Argument(format!(
            "no file in recipe group `{group}` declares a `recipe_description`; \
             one of them must carry it and the rest inherit it (VAULT.md §8)"
        )));
    }
    for (file, record) in files.iter().zip(&records) {
        validate(record).map_err(|err| KgError::FileFormat {
            path: file.clone(),
            message: err.to_string(),
        })?;
    }

    let mut written = Vec::with_capacity(records.len());
    for record in records {
        set(graph, &record)?;
        written.push((record.recipe, record.name));
    }
    Ok(written)
}

/// Without the `okf` feature there is no frontmatter reader, so the markdown
/// dialect is unavailable rather than silently empty.
#[cfg(not(feature = "okf"))]
fn import_markdown_dir(
    _graph: &mut DirGraph,
    path: &Path,
) -> Result<Vec<(String, String)>, KgError> {
    Err(KgError::FileFormat {
        path: path.to_path_buf(),
        message: "markdown recipe files need the `okf` feature; this build reads \
                  .json catalogues only"
            .to_string(),
    })
}

/// Write [`export_value`] to `path` as pretty-printed JSON.
pub fn export_path(graph: &DirGraph, path: &Path) -> Result<(), KgError> {
    let text = serde_json::to_string_pretty(&export_value(graph))
        .map_err(|error| KgError::Argument(error.to_string()))?;
    std::fs::write(path, text).map_err(KgError::FileIo)
}

#[cfg(test)]
#[path = "records_tests.rs"]
mod tests;
