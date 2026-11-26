//! Write a graph back out as an Obsidian vault (VAULT.md §10) — the inverse of
//! [`crate::okf::build`].
//!
//! The reader's rules run backwards here: a note's label decides its folder so
//! the label ladder recovers it without a `type:` key, its frontmatter is the
//! properties the reader turned into columns, and its outgoing edges become the
//! wikilink-valued keys §4.3 turns back into edges. Everything the build
//! *synthesized* — `Folder`, hub, attachment and stub nodes — is left out,
//! because the next import regenerates it.
//!
//! Nothing here overwrites a file it did not write. `.kglite/export-manifest.json`
//! records a hash per exported file; a file missing from it, or one whose bytes
//! have moved since, belongs to a human and is refused rather than replaced.

use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;
use crate::graph::DirGraph;
use crate::okf::model::{
    Profile, ATTACHMENT_LABEL, DEFAULT_BODY_PROPERTY, FOLDER_LABEL, IMAGE_LABEL, SOURCE_LABEL,
    TAG_LABEL,
};
use crate::okf::vault_config::{CONFIG_DIR, RECIPES_DIR, SKILLS_DIR};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

mod edge_tables;
mod manifest;
mod paths;
mod yaml_out;

pub use manifest::{MANIFEST_FILE, MANIFEST_VERSION};

use manifest::Writer;
use paths::{assign_paths, sanitize_segment};
use yaml_out::{lower_snake, render_frontmatter, Tree};

/// Properties the exporter never writes into frontmatter: the three columns the
/// reader derives from the file itself, and the stub marker. The prose is
/// removed separately, under whatever name `body:` gave it. Embeddings are not
/// properties at all — they live in the graph's own store — so nothing here
/// has to exclude them.
const NEVER_IN_FRONTMATTER: [&str; 4] = ["concept_id", "title", "file_path", "_provisional"];

/// Labels the build synthesizes from other nodes' content (VAULT.md §10.1).
/// None of them is a file: the next import makes them again from the notes.
const SYNTHESIZED_LABELS: [&str; 5] = [
    TAG_LABEL,
    SOURCE_LABEL,
    FOLDER_LABEL,
    IMAGE_LABEL,
    ATTACHMENT_LABEL,
];

/// What [`export`] may do to the target directory.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Overwrite files the manifest does not own, or that a human has edited
    /// since the last export, and delete owned files that have been edited.
    /// Without it each of those is refused and counted.
    pub force: bool,
    /// The directory the graph's attachments were read from, so their bytes can
    /// be copied into the exported vault. `None` falls back to the graph's own
    /// `source_root` provenance stamp (VAULT.md §12) — a vault-built graph
    /// knows where its pictures are, and asking the caller to repeat a path
    /// the graph carries is how the two come to disagree. With neither, every
    /// body reference is left pointing at a file the export did not write, and
    /// counted as unresolved.
    pub source_root: Option<PathBuf>,
    /// The property holding each note's prose — `.kglite/vault.yaml`'s `body:`
    /// (VAULT.md §7), which a graph does not carry, so the caller repeats it.
    pub body_property: String,
    /// Edge type → the heading whose table its edges are written under
    /// (VAULT.md §7.3, §10.6), for a graph with no vault to declare it — or to
    /// override what one declared. Merged **over** the `export.edge_tables:`
    /// of the source vault's own `.kglite/vault.yaml`, per type: an entry here
    /// wins, and a type only the vault names still gets its table.
    pub edge_tables: BTreeMap<String, String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            force: false,
            source_root: None,
            body_property: DEFAULT_BODY_PROPERTY.to_string(),
            edge_tables: BTreeMap::new(),
        }
    }
}

/// What an export did, and what it declined to do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExportReport {
    /// Files created or replaced.
    pub files_written: usize,
    /// Files whose bytes already matched what this export would write, so they
    /// were left alone — their modification times do not move.
    pub files_unchanged: usize,
    /// Manifest-owned files whose node is gone from the graph, removed.
    pub files_deleted: usize,
    /// Writes and deletions declined for safety; [`ExportReport::refusals`]
    /// says which, and why.
    pub files_refused: usize,
    /// One line per refusal, in path order.
    pub refusals: Vec<String>,
    /// Edge properties (`section`, `anchor`, `alt`, `ordinal`) dropped —
    /// frontmatter lists carry targets, not properties (VAULT.md §10.9).
    pub edge_properties_dropped: usize,
    /// Attachment files copied in from `source_root`.
    pub attachments_copied: usize,
    /// Attachment nodes whose bytes could not be copied, because no
    /// `source_root` was given or the file is not under it. Their body
    /// references are written as they stand and resolve to nothing.
    pub attachments_unresolved: usize,
    /// `.kglite/skills/*.md` files the graph's `KgliteSkill` nodes produced.
    pub skills_written: usize,
    /// `.kglite/recipes/*.md` files the graph's `KgliteRecipe` nodes produced.
    pub recipes_written: usize,
    /// What the export could not do as declared, in declaration order — a
    /// declared edge table the source vault has no import rule for, or one no
    /// exported note emits. Nothing here failed; each line is a table the
    /// author will not get back (VAULT.md §10.6).
    pub warnings: Vec<String>,
}

impl ExportReport {
    /// The report as the text `kglite okf export` prints.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("files written: {}\n", self.files_written));
        out.push_str(&format!("files unchanged: {}\n", self.files_unchanged));
        out.push_str(&format!("files deleted: {}\n", self.files_deleted));
        out.push_str(&format!("files refused: {}\n", self.files_refused));
        out.push_str(&format!(
            "attachments copied: {} (unresolved: {})\n",
            self.attachments_copied, self.attachments_unresolved
        ));
        out.push_str(&format!(
            "skills: {}, recipes: {}\n",
            self.skills_written, self.recipes_written
        ));
        out.push_str(&format!(
            "edge properties dropped: {}\n",
            self.edge_properties_dropped
        ));
        for line in &self.warnings {
            out.push_str(&format!("warning: {line}\n"));
        }
        if self.refusals.is_empty() {
            out.push_str("refusals: none\n");
        } else {
            out.push_str("refusals:\n");
            for line in &self.refusals {
                out.push_str(&format!("  - {line}\n"));
            }
        }
        out
    }
}

/// One node on its way to a file.
pub(super) struct Note {
    idx: NodeIndex,
    /// The node's label — its output folder, and the rung the next import's
    /// label ladder recovers it from (VAULT.md §2.1).
    pub(super) label: String,
    id: String,
    title: String,
    /// The `file_path` the build stored, when this graph came from a vault.
    pub(super) file_path: Option<String>,
    props: Vec<(String, Value)>,
    body: Option<String>,
    /// Vault-relative output path, assigned by [`paths::assign_paths`].
    pub(super) out: String,
}

impl Note {
    /// The output file's stem — what a `[[wikilink]]` to this note names.
    fn stem(&self) -> &str {
        let file = self.out.rsplit('/').next().unwrap_or(&self.out);
        file.strip_suffix(".md").unwrap_or(file)
    }

    /// The output path minus `.md` — the folder-qualified form §5.1 resolves
    /// as a vault-relative id when a bare stem is ambiguous.
    fn qualified(&self) -> &str {
        self.out.strip_suffix(".md").unwrap_or(&self.out)
    }

    /// The name the file gets when its path is not preserved: the title if
    /// there is one, else the id (VAULT.md §10.2).
    pub(super) fn display_name(&self) -> &str {
        if self.title.is_empty() {
            &self.id
        } else {
            &self.title
        }
    }

    /// The id, for the `-<id>` suffix a case-insensitive collision appends.
    pub(super) fn id(&self) -> &str {
        &self.id
    }
}

/// Write `graph` into `dir` as a vault (VAULT.md §10).
///
/// The directory is created if it does not exist. An existing one is written
/// *into*: only files this export owns — the ones its manifest records — are
/// replaced or removed, and everything else is left exactly as it is unless
/// [`ExportOptions::force`] says otherwise.
///
/// `Err` is reserved for a target that cannot be used at all (a path that is
/// not a directory, an unreadable manifest, a write that fails). A file the
/// export declined to touch is a refusal *in* the report, because a vault of a
/// thousand notes must not fail wholesale over one edited file.
pub fn export(graph: &DirGraph, dir: &Path, opts: &ExportOptions) -> Result<ExportReport, String> {
    if dir.exists() && !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    let Collected {
        mut notes,
        attachments,
        stubs,
    } = collect(graph, &opts.body_property);
    assign_paths(&mut notes);
    let index = LinkIndex::new(&notes);
    let (declared, mut warnings) = declared_edge_tables(graph, opts);
    let (edges, edge_properties_dropped) = outgoing_edges(graph, &notes, &stubs, &declared);
    warnings.extend(unemitted_edge_tables(&declared, &edges));

    let mut writer = Writer::open(dir, opts.force)?;
    writer.report.edge_properties_dropped = edge_properties_dropped;
    writer.report.warnings = warnings;
    for note in &notes {
        let text = render_note(note, &notes, &index, &edges, &declared);
        writer.put(&note.out.clone(), text.as_bytes())?;
    }
    write_carried(graph, &mut writer)?;
    // The caller's root wins; the graph's own provenance stands in for it.
    let source_root = opts
        .source_root
        .clone()
        .or_else(|| graph.source_root.as_ref().map(PathBuf::from));
    copy_attachments(&attachments, source_root.as_deref(), &mut writer)?;
    writer.finish()
}

/// Read every node the graph offers into the two lists the export works from:
/// the notes that become files, and the attachments whose bytes are copied.
///
/// A graph built from a vault carries `file_path` on every note and on nothing
/// else; a graph built any other way carries it nowhere. Which of the two this
/// is decides what a node *without* the property means — a hub node the build
/// synthesized, or an ordinary node of a graph that simply never had files.
fn collect(graph: &DirGraph, body_property: &str) -> Collected {
    let skill_label = crate::graph::skills::SKILL_LABEL;
    let recipe_label = crate::graph::recipes::RECIPE_LABEL;
    let mut attachments = Vec::new();
    let mut stubs: HashMap<NodeIndex, String> = HashMap::new();
    let mut vault_built = false;
    let mut candidates: Vec<Note> = Vec::new();

    for idx in graph.graph.node_indices() {
        let Some(view) = graph.node_view(idx) else {
            continue;
        };
        let label = view.node_type_str(&graph.interner).to_string();
        let provisional = matches!(
            view.get_property_value("_provisional"),
            Some(Value::Boolean(true))
        );
        if label == IMAGE_LABEL || label == ATTACHMENT_LABEL {
            // The attachment node's id field *is* its vault-relative path
            // (VAULT.md §6.3), so it is read as an id, not as a property.
            if !provisional {
                if let Value::String(path) = view.id().as_ref() {
                    attachments.push(path.clone());
                }
            }
            continue;
        }
        if provisional {
            // Not a file (§10.1), but still a link *target*: an edge naming a
            // stub is a dangling link, and writing `[[<name>]]` for it is what
            // makes the next import dangle in the same place. A stub only
            // reachable from the body needs nothing — the prose already says
            // it — but one that came from a frontmatter key has no other
            // spelling to survive in.
            if let Value::String(name) = view.id().as_ref() {
                stubs.insert(idx, name.clone());
            }
            continue;
        }
        if SYNTHESIZED_LABELS.contains(&label.as_str())
            || label == skill_label
            || label == recipe_label
        {
            continue;
        }
        let file_path = match view.get_property_value("file_path") {
            Some(Value::String(p)) if !p.is_empty() => Some(p),
            _ => None,
        };
        if file_path.is_some() {
            vault_built = true;
        }
        let mut props = view.property_pairs_named(&graph.interner);
        props.retain(|(k, v)| {
            !NEVER_IN_FRONTMATTER.contains(&k.as_str()) && !matches!(v, Value::Null)
        });
        props.sort_by(|a, b| a.0.cmp(&b.0));
        let body = props
            .iter()
            .position(|(k, _)| k == body_property)
            .map(|at| props.remove(at).1)
            .map(|v| match v {
                Value::String(s) => s,
                other => crate::datatypes::values::raw_string(&other),
            });
        candidates.push(Note {
            idx,
            label,
            id: scalar_string(&view.id()),
            title: scalar_string(&view.title()),
            file_path,
            props,
            body,
            out: String::new(),
        });
    }

    let mut notes: Vec<Note> = candidates
        .into_iter()
        .filter(|note| !vault_built || note.file_path.is_some())
        .collect();
    notes.sort_by(|a, b| a.label.cmp(&b.label).then_with(|| a.id.cmp(&b.id)));
    attachments.sort();
    attachments.dedup();
    Collected {
        notes,
        attachments,
        stubs,
    }
}

/// What one pass over the graph's nodes found.
struct Collected {
    notes: Vec<Note>,
    /// Vault-relative paths of the attachment files, sorted and de-duplicated.
    attachments: Vec<String>,
    /// `_provisional` stub nodes by index, with the name a link to one writes.
    stubs: HashMap<NodeIndex, String>,
}

/// A node's id or title as the string the format writes. A non-string one is
/// rendered rather than dropped: the file has to be named something, and a
/// numeric id is a legitimate thing for a producer to have written.
fn scalar_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => crate::datatypes::values::raw_string(other),
    }
}

/// The stem ambiguity §5.1 asks a converter to qualify away: a wikilink whose
/// bare stem two exported notes share has to name the folder too.
struct LinkIndex {
    /// Lowercased output stem → how many notes carry it.
    stem_uses: HashMap<String, usize>,
}

impl LinkIndex {
    fn new(notes: &[Note]) -> Self {
        let mut stem_uses: HashMap<String, usize> = HashMap::new();
        for note in notes {
            *stem_uses
                .entry(note.stem().to_ascii_lowercase())
                .or_default() += 1;
        }
        LinkIndex { stem_uses }
    }

    /// The wikilink body for a target: its bare stem, or the folder-qualified
    /// path when another exported note shares that stem.
    fn wikilink(&self, note: &Note) -> String {
        if self.stem_uses.get(&note.stem().to_ascii_lowercase()) > Some(&1) {
            note.qualified().to_string()
        } else {
            note.stem().to_string()
        }
    }
}

/// One outgoing edge, reduced to what the export can write: a target, and —
/// for a type `export.edge_tables` declares — the properties its row carries.
/// An undeclared type's properties are dropped and counted (VAULT.md §10.9),
/// so they are not carried here at all.
struct OutEdge {
    conn_type: String,
    target: Target,
    props: Vec<(String, Value)>,
}

/// What a frontmatter wikilink can point at: a note the export writes, or the
/// name of a stub it does not.
enum Target {
    Note(usize),
    Stub(String),
}

/// Every edge between two exported notes, grouped by source, and the number of
/// edge properties the export drops on the way (VAULT.md §10.9).
///
/// There is no list of "structural" types here on purpose: the *endpoints*
/// decide it. `CONTAINS` leaves a `Folder`, `TAGGED` and every hub edge reach a
/// hub node, `HAS_IMAGE` and `HAS_ATTACHMENT` reach an attachment, and none of
/// those is a file, so none of them can be written as a wikilink. Naming the
/// types instead would also drop a note-to-note `CONTAINS` in a graph that was
/// never a vault, where nothing else expresses it.
///
/// The count covers every edge the export *writes* whose type no
/// `export.edge_tables` entry declares: an edge to a node the export does not
/// write loses nothing it could have carried, and a declared type's properties
/// are written rather than lost (VAULT.md §7.3, §10.9).
fn outgoing_edges(
    graph: &DirGraph,
    notes: &[Note],
    stubs: &HashMap<NodeIndex, String>,
    declared: &BTreeMap<String, String>,
) -> (HashMap<NodeIndex, Vec<OutEdge>>, usize) {
    let positions: HashMap<NodeIndex, usize> = notes
        .iter()
        .enumerate()
        .map(|(at, note)| (note.idx, at))
        .collect();
    let mut out: HashMap<NodeIndex, Vec<OutEdge>> = HashMap::new();
    let mut dropped = 0usize;
    for edge in graph.graph.edge_indices() {
        let Some((src, tgt)) = graph.graph.edge_endpoints(edge) else {
            continue;
        };
        if !positions.contains_key(&src) {
            continue;
        }
        let Some(data) = graph.graph.edge_weight(edge) else {
            continue;
        };
        // A target the export does not write has no wikilink to name it: hub
        // nodes, `Source` URLs, attachments and stubs all land here, which is
        // how `TAGGED`, `HAS_KEYWORD`, `HAS_IMAGE` and `HAS_ATTACHMENT` leave
        // the frontmatter without being named one by one.
        let target = match positions.get(&tgt) {
            Some(&at) => Target::Note(at),
            None => match stubs.get(&tgt) {
                Some(name) => Target::Stub(name.clone()),
                None => continue,
            },
        };
        // Counted only for an edge the export *writes*: an edge to a node it
        // does not write loses nothing it could have carried. Counting before
        // this match made every `HAS_SECTION` a dropped property the moment a
        // vault declared `structure:` (VAULT.md §7.1, §10.9).
        let conn_type = data.connection_type_str(&graph.interner).to_string();
        // A declared type's properties are written, as a table (§10.6), so
        // they are never part of the count: where the export leaves such an
        // edge out because the body already states it, the prose carries them
        // instead — the same reason a body-stated edge keeps them today.
        let props = match declared.contains_key(&conn_type) {
            true => data
                .property_iter(&graph.interner)
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect(),
            false => {
                dropped += data.property_count();
                Vec::new()
            }
        };
        out.entry(src).or_default().push(OutEdge {
            conn_type,
            target,
            props,
        });
    }
    (out, dropped)
}

/// The edge types written as body tables, and the heading each goes under
/// (VAULT.md §7.3, §10.6), with what the caller should know about them.
///
/// Two sources. The **source vault's own** `.kglite/vault.yaml`, found through
/// the `source_root` the caller named or the graph's provenance stamp — the
/// declaration lives with the vault because the same file carries the
/// `structure.tables:` rule that reads the table back. And
/// [`ExportOptions::edge_tables`], which is how a graph that never was a vault
/// declares one; it is merged **over** the file, per type, so a caller can add
/// a type or move one type's heading without restating the rest.
///
/// A `vault.yaml` that will not parse is a warning rather than an error: the
/// export is writing a *different* directory, and one unreadable file beside
/// the graph's source is no reason to refuse a vault of a thousand notes.
fn declared_edge_tables(
    graph: &DirGraph,
    opts: &ExportOptions,
) -> (BTreeMap<String, String>, Vec<String>) {
    let mut warnings = Vec::new();
    let root = opts
        .source_root
        .clone()
        .or_else(|| graph.source_root.as_ref().map(PathBuf::from));
    let config = match root.as_deref().map(crate::okf::vault_config::load) {
        Some(Ok(config)) => config,
        Some(Err(reason)) => {
            warnings.push(format!(
                "{reason}; no `export.edge_tables` were read from the source vault"
            ));
            None
        }
        None => None,
    };
    let mut declared = config
        .as_ref()
        .map(|config| config.export_edge_tables.clone())
        .unwrap_or_default();
    declared.extend(opts.edge_tables.clone());
    if let Some(config) = &config {
        warnings.extend(unreadable_edge_tables(&declared, config));
    }
    (declared, warnings)
}

/// A declared table the source vault cannot read back (VAULT.md §10.6).
///
/// No export writes a `vault.yaml` (§10.9 loss 4), so the table travels and the
/// rule that reads it does not: a vault whose own `structure.tables:` has no
/// `edges: true` rule for the type under that heading exports a table its next
/// build reads as prose. The author hears it here rather than from a missing
/// edge three steps later.
fn unreadable_edge_tables(
    declared: &BTreeMap<String, String>,
    config: &crate::okf::vault_config::VaultConfig,
) -> Vec<String> {
    declared
        .iter()
        .filter(|(conn_type, heading)| {
            !config.structure.as_ref().is_some_and(|structure| {
                structure.tables.iter().any(|rule| {
                    rule.edges && &&rule.edge == conn_type && rule.under_heading.is_match(heading)
                })
            })
        })
        .map(|(conn_type, heading)| {
            format!(
                "`export.edge_tables.{conn_type}` writes a table under `{heading}`, but the \
                 source vault declares no `structure.tables:` rule with `edges: true`, \
                 `edge: {conn_type}` and an `under_heading:` matching that heading — the \
                 exported table reads back as prose (VAULT.md §7.1, §10.6)"
            )
        })
        .collect()
}

/// A declared type no exported note emits (VAULT.md §10.6).
///
/// The common cause is a type whose *source* is a node the export does not
/// write — every node `structure:` derives is part of a note's prose and has no
/// file (§10.1), so an edge leaving one has nowhere to be written from.
fn unemitted_edge_tables(
    declared: &BTreeMap<String, String>,
    edges: &HashMap<NodeIndex, Vec<OutEdge>>,
) -> Vec<String> {
    declared
        .keys()
        .filter(|conn_type| {
            !edges
                .values()
                .flatten()
                .any(|edge| &&edge.conn_type == conn_type)
        })
        .map(|conn_type| {
            format!(
                "`export.edge_tables.{conn_type}`: no exported source note emits {conn_type}, \
                 so no table is written — an edge leaving a node the export does not write \
                 has no note to write it in (VAULT.md §10.1)"
            )
        })
        .collect()
}

/// The note's prose with every declared type's table written into it (§10.6).
///
/// A type is *in play* for a note when the note has an edge of it, or when the
/// body already carries its heading: the second because a table whose edges are
/// gone from the graph has to go with them, or the next import would make them
/// again.
fn declared_tables(
    note: &Note,
    outgoing: &[OutEdge],
    notes: &[Note],
    index: &LinkIndex,
    declared: &BTreeMap<String, String>,
) -> String {
    let body = note.body.as_deref().unwrap_or("");
    let in_play: Vec<(&str, &str)> = declared
        .iter()
        .map(|(conn_type, heading)| (conn_type.as_str(), heading.as_str()))
        .filter(|(conn_type, heading)| {
            outgoing.iter().any(|edge| edge.conn_type == *conn_type) || body.contains(*heading)
        })
        .collect();
    if in_play.is_empty() {
        return body.to_string();
    }
    let headings: Vec<&str> = in_play.iter().map(|(_, heading)| *heading).collect();
    // What the note states *outside* the tables the exporter owns: an edge one
    // of those states keeps its properties in that prose, and a row for it
    // would make a second edge on the next import.
    let stated = body_links(
        note,
        &edge_tables::prose_outside_owned_tables(body, &headings),
    );
    let tables: Vec<(&str, Vec<edge_tables::Row>)> = in_play
        .iter()
        .map(|(conn_type, heading)| {
            let rows = edge_tables::rows_for(outgoing, conn_type, notes, index, |edge| {
                body_states_edge(&stated, edge, notes)
            });
            (*heading, rows)
        })
        .collect();
    edge_tables::apply(body, &tables)
}

/// Render one note: frontmatter, then the body verbatim (VAULT.md §10.3–§10.6).
fn render_note(
    note: &Note,
    notes: &[Note],
    index: &LinkIndex,
    edges: &HashMap<NodeIndex, Vec<OutEdge>>,
    declared: &BTreeMap<String, String>,
) -> String {
    let outgoing: &[OutEdge] = edges.get(&note.idx).map_or(&[], Vec::as_slice);
    // The declared tables are written into the prose first, so the frontmatter
    // pass below reads the body the file will actually carry — a table's own
    // `[[cells]]` are links like any other, and an edge one of them states is
    // an edge the frontmatter must not state again (§10.6).
    let body = declared_tables(note, outgoing, notes, index, declared);
    let mut tree = Tree::default();
    // `type:` is never emitted (§10.4): the folder carries the label, so
    // writing it too would make a later folder move a no-op.
    if note.id != note.stem() {
        tree.insert("id", Value::String(note.id.clone()));
    }
    if !note.title.is_empty() && note.title != recovered_title(note, &body) {
        tree.insert("title", Value::String(note.title.clone()));
    }
    for (key, value) in &note.props {
        tree.insert(key, value.clone());
    }
    for (key, targets) in edge_keys(note, &body, notes, index, outgoing, declared) {
        tree.insert_wikilinks(&key, targets);
    }

    let front = render_frontmatter(&tree);
    let body = body.as_str();
    match (front.is_empty(), body.is_empty()) {
        (true, true) => String::new(),
        (true, false) => ensure_newline(body),
        (false, true) => format!("---\n{front}---\n"),
        // No separator line between the closing `---` and the prose: the
        // reader keeps whatever follows the terminator *as* the body, so a
        // blank line written here comes back as a leading newline on the
        // property and the next export writes another one. A note whose author
        // left a blank line there still has it, in the body, and gets it back.
        (false, false) => format!("---\n{front}---\n{}", ensure_newline(body)),
    }
}

fn ensure_newline(text: &str) -> String {
    if text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{text}\n")
    }
}

/// The wikilink-valued frontmatter keys this note's edges become, sorted by key
/// and by target within each key (VAULT.md §10.6, §10.8).
fn edge_keys(
    note: &Note,
    body: &str,
    notes: &[Note],
    index: &LinkIndex,
    outgoing: &[OutEdge],
    declared: &BTreeMap<String, String>,
) -> BTreeMap<String, Vec<String>> {
    if outgoing.is_empty() {
        return BTreeMap::new();
    }
    let mentioned = body_links(note, body);
    let mut by_key: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for edge in outgoing {
        // A declared type is written as a table in the body, never as a
        // frontmatter list: writing both would make the edge twice (§10.6).
        if declared.contains_key(&edge.conn_type) || body_states_edge(&mentioned, edge, notes) {
            continue;
        }
        let name = match &edge.target {
            Target::Note(at) => index.wikilink(&notes[*at]),
            Target::Stub(name) => name.clone(),
        };
        by_key
            .entry(lower_snake(&edge.conn_type))
            .or_default()
            .insert(name);
    }
    by_key
        .into_iter()
        .map(|(key, targets)| (key, targets.into_iter().collect()))
        .collect()
}

/// The title the next import will read off the file this export is writing:
/// §3's ladder run forwards over the bytes going out — a `name:` property, the
/// body's first heading, then the filename stem.
///
/// `title:` is emitted exactly when the note's own title is not that (§10.3).
/// Comparing against the *stem* alone is not enough: the heading rung sits
/// above it, so a note titled after its file but opening with a heading came
/// back titled by the heading.
fn recovered_title(note: &Note, body: &str) -> String {
    let named = note
        .props
        .iter()
        .find(|(k, _)| k == "name")
        .and_then(|(_, v)| match v {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            Value::Null | Value::String(_) => None,
            other => Some(crate::datatypes::values::raw_string(other)),
        });
    named
        .or_else(|| crate::okf::first_heading(body))
        .unwrap_or_else(|| note.stem().to_string())
}

/// Every link the note's own prose states, as `(edge type, lowercased target)`.
/// Re-extracted with the reader's own scanner, so "already in the body" means
/// exactly what the next import will read there — including the type the
/// heading ladder (§5.3) gives it, which is why a link under `## Related` is
/// not a `LINKS_TO`.
fn body_links(note: &Note, body: &str) -> Vec<(String, String)> {
    if body.is_empty() {
        return Vec::new();
    }
    let dir = match note.out.rfind('/') {
        Some(at) => &note.out[..at],
        None => "",
    };
    crate::okf::links::extract(body, dir, &Profile::obsidian())
        .links
        .into_iter()
        .map(|link| {
            (
                link.conn_type,
                link.target.trim_end_matches(".md").to_ascii_lowercase(),
            )
        })
        .collect()
}

/// Whether the prose already states *this* edge — the same type to the same
/// target (VAULT.md §10.6).
///
/// The type has to match. A `RELATED_TO` between two notes the body also links
/// plainly is a different statement from that link, and dropping it would
/// retype the edge to `LINKS_TO` on the next import; conversely an edge the
/// body does state, of the type the body gives it, is a duplicate, and writing
/// it into frontmatter as well makes a second edge — one with the body's
/// `section` and one without.
fn body_states(
    written: &[(String, String)],
    conn_type: &str,
    names_target: impl Fn(&str) -> bool,
) -> bool {
    written
        .iter()
        .any(|(conn, target)| conn == conn_type && names_target(target))
}

/// [`body_states`] for one outgoing edge, whichever kind of target it has.
fn body_states_edge(written: &[(String, String)], edge: &OutEdge, notes: &[Note]) -> bool {
    match &edge.target {
        Target::Note(at) => body_states(written, &edge.conn_type, |name| names(&notes[*at], name)),
        Target::Stub(name) => {
            let lowered = name.to_ascii_lowercase();
            body_states(written, &edge.conn_type, |written| written == lowered)
        }
    }
}

/// Whether one of the body's targets names this note — by stem, by id, by its
/// vault-relative path, by title or by one of its `aliases:`, which are the
/// rungs §5.2 resolves on.
fn names(note: &Note, target: &str) -> bool {
    if target == note.stem().to_ascii_lowercase()
        || target == note.id.to_ascii_lowercase()
        || target == note.qualified().to_ascii_lowercase()
        || (!note.title.is_empty() && target == note.title.to_ascii_lowercase())
    {
        return true;
    }
    note.props
        .iter()
        .find(|(k, _)| k == "aliases")
        .is_some_and(|(_, v)| match v {
            Value::List(items) => items.iter().any(|item| match item {
                Value::String(s) => s.to_ascii_lowercase() == target,
                _ => false,
            }),
            _ => false,
        })
}

/// The graph's skills and recipes as `.kglite/` markdown (VAULT.md §8).
fn write_carried(graph: &DirGraph, writer: &mut Writer) -> Result<(), String> {
    for summary in crate::graph::skills::list(graph) {
        let record = crate::graph::skills::get(graph, &summary.name)
            .map_err(|e| format!("reading skill {}: {e}", summary.name))?;
        let path = format!(
            "{CONFIG_DIR}/{SKILLS_DIR}/{}.md",
            sanitize_segment(&record.name)
        );
        let text = crate::graph::skills::render_markdown(&record);
        writer.put(&path, text.as_bytes())?;
        writer.report.skills_written += 1;
    }
    // One file per query, named `<group>.<query>.md`: the group id alone is not
    // unique, and the import reads the ids from the frontmatter rather than
    // from the filename, so the name only has to be stable and distinct.
    for record in crate::graph::recipes::list(graph) {
        let path = format!(
            "{CONFIG_DIR}/{RECIPES_DIR}/{}.{}.md",
            sanitize_segment(&record.recipe),
            sanitize_segment(&record.name)
        );
        let text = crate::graph::recipes::render_markdown(&record);
        writer.put(&path, text.as_bytes())?;
        writer.report.recipes_written += 1;
    }
    Ok(())
}

/// Copy each attachment's bytes from the source root (VAULT.md §10.9).
///
/// `root` is the caller's `source_root`, or the graph's own provenance stamp
/// where the caller named none. Without either there is nothing to copy: the
/// body reference is left as the author wrote it and counted, so a caller
/// knows the vault it just wrote is missing its figures.
///
/// The copy carries the source file's modification time, because §6.3 makes
/// `mtime` a node property read off `stat` and §10.9 does not list it among
/// the losses — see [`Writer::put_at`]. A source whose metadata cannot be read
/// leaves the destination with the time of the write, which is the best
/// available answer rather than a wrong one.
fn copy_attachments(
    attachments: &[String],
    root: Option<&Path>,
    writer: &mut Writer,
) -> Result<(), String> {
    let Some(root) = root else {
        writer.report.attachments_unresolved += attachments.len();
        return Ok(());
    };
    for rel in attachments {
        let from = root.join(rel);
        match std::fs::read(&from) {
            Ok(bytes) => {
                let modified = std::fs::metadata(&from)
                    .ok()
                    .and_then(|m| m.modified().ok());
                writer.put_at(rel, &bytes, modified)?;
                writer.report.attachments_copied += 1;
            }
            Err(_) => writer.report.attachments_unresolved += 1,
        }
    }
    Ok(())
}

#[cfg(test)]
mod export_tests;
#[cfg(test)]
mod roundtrip_tests;
