//! The round-trip contract (VAULT.md §10.9) as a differential corpus.
//!
//! Two properties, over a list of vaults rather than one fixture:
//!
//! - **`export ∘ import ∘ export`** is byte-identical: exporting a graph,
//!   reading the vault back and exporting *that* produces the same tree, file
//!   for file and byte for byte, the manifest included.
//! - **`import ∘ export ∘ import ≡ import`**, apart from §10.9's documented
//!   losses.
//!
//! The losses are taken **once**, on the way out of the author's vault, so the
//! two are asserted together by running three rounds: every vault's second and
//! third export must be byte-identical and their graphs equal — a loss that
//! kept happening would show up there as unbounded drift — and a vault taking
//! no loss that moves bytes must be identical from the *first* export. Then
//! each loss §10.9 names is asserted below as a difference between the first
//! graph and the second that *must* exist. A loss the spec claims and the code
//! does not take is a wrong spec, so the two refinements this phase found
//! ([`a_declared_int_survives_the_round_trip`],
//! [`a_body_stated_edge_keeps_its_properties`]) are asserted as survivals, and
//! §10.9 was rewritten to say so.
//!
//! [`Shape`] is what "graph-equal" means here: the label counter, the
//! edge-type counter, the per-type edge-property counter and every node's
//! property map, rendered to strings so two independently built graphs compare
//! by value.

use super::*;
use crate::okf::model::{BuildOptions, BuildReport, Dialect};
use std::collections::BTreeMap;
use std::sync::Arc;

// ── the harness ────────────────────────────────────────────────────────────

fn golden_vault() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/okf/golden/vault")
}

fn vault_opts() -> BuildOptions {
    BuildOptions::for_dialect(Dialect::Obsidian)
}

fn build_vault(dir: &Path) -> Arc<DirGraph> {
    crate::okf::build(dir, &vault_opts()).unwrap().graph
}

fn write_vault(root: &Path, files: &[(&str, &str)]) {
    for (rel, text) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

/// A vault of `(relative path, contents)` pairs under its own temp directory.
fn vault_of(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_vault(dir.path(), files);
    dir
}

/// Every file's bytes under `dir`, vault-relative — the whole tree as one
/// comparable value, `.kglite/export-manifest.json` included.
fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    collect_files(dir, dir, &mut out);
    out
}

fn collect_files(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_files(root, &path, out);
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(rel, std::fs::read(&path).unwrap());
        }
    }
}

/// A graph reduced to what two independent builds of one vault must agree on.
#[derive(Debug, Default, PartialEq, Eq)]
struct Shape {
    labels: BTreeMap<String, usize>,
    edges: BTreeMap<String, usize>,
    /// Edge properties carried, per edge type — §10.9's first loss is measured
    /// here, and so is the prose re-derivation that undoes it.
    edge_props: BTreeMap<String, usize>,
    nodes: BTreeMap<(String, String), BTreeMap<String, String>>,
}

impl Shape {
    fn of(graph: &DirGraph) -> Shape {
        let mut shape = Shape::default();
        for idx in graph.graph.node_indices() {
            let Some(view) = graph.node_view(idx) else {
                continue;
            };
            let label = view.node_type_str(&graph.interner).to_string();
            *shape.labels.entry(label.clone()).or_default() += 1;
            let id = crate::datatypes::values::raw_string(&view.id());
            let props: BTreeMap<String, String> = view
                .property_pairs_named(&graph.interner)
                .into_iter()
                .filter(|(_, v)| !matches!(v, Value::Null))
                .map(|(k, v)| (k, crate::datatypes::values::raw_string(&v)))
                .collect();
            shape.nodes.insert((label, id), props);
        }
        for edge in graph.graph.edge_indices() {
            if let Some(data) = graph.graph.edge_weight(edge) {
                let conn = data.connection_type_str(&graph.interner).to_string();
                *shape.edges.entry(conn.clone()).or_default() += 1;
                *shape.edge_props.entry(conn).or_default() += data.properties.len();
            }
        }
        shape
    }

    fn label(&self, label: &str) -> usize {
        self.labels.get(label).copied().unwrap_or(0)
    }

    fn edge(&self, conn: &str) -> usize {
        self.edges.get(conn).copied().unwrap_or(0)
    }

    /// One node's properties, or `None` when no node carries that identity.
    fn props(&self, label: &str, id: &str) -> Option<&BTreeMap<String, String>> {
        self.nodes.get(&(label.to_string(), id.to_string()))
    }
}

/// Three rounds of `export ∘ import`, from a vault or from a bare graph: the
/// trees each export wrote and the graph each import built.
///
/// The temp directories are kept alive by the value — dropping one would
/// remove the tree its snapshot was taken from.
struct Trip {
    first: Shape,
    second: Shape,
    third: Shape,
    /// The three exported trees, in order.
    exports: Vec<BTreeMap<String, Vec<u8>>>,
    report: ExportReport,
    /// The build report of the re-import — where the `vault.yaml` losses show.
    second_report: BuildReport,
    _dirs: Vec<tempfile::TempDir>,
}

impl Trip {
    /// Run the trip from a source vault, copying its attachments in.
    fn from_vault(source: &Path) -> Trip {
        Trip::run(source, true, false)
    }

    /// Run it with nothing naming the source root — no option, and no
    /// provenance stamp on the graph either — so no attachment bytes travel.
    fn rootless(source: &Path) -> Trip {
        Trip::run(source, false, false)
    }

    /// Run it with the source's own `.kglite/vault.yaml` copied into each
    /// exported tree before it is read back.
    ///
    /// No export writes that file (§10.9 loss 4), so a vault whose declarations
    /// *are* the contract — `structure:` deriving nodes, and the
    /// `structure.tables … edges: true` rule that reads a declared edge table
    /// back (§10.6) — cannot round-trip without it. Copying it is what the
    /// spec asks an author to do, done here.
    fn carrying_config(source: &Path) -> Trip {
        Trip::run(source, true, true)
    }

    fn run(source: &Path, with_root: bool, carry_config: bool) -> Trip {
        let mut graph = (*build_vault(source)).clone();
        if !with_root {
            graph.source_root = None;
        }
        let first = Shape::of(&graph);
        let root = with_root.then(|| source.to_path_buf());
        let config = carry_config.then(|| crate::okf::vault_config::config_path(source));
        Trip::from_graph(&graph, root, first, config)
    }

    fn from_graph(
        graph: &DirGraph,
        source_root: Option<PathBuf>,
        first: Shape,
        config: Option<PathBuf>,
    ) -> Trip {
        let carry_root = source_root.is_some();
        let dirs: Vec<tempfile::TempDir> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let mut root = source_root;
        let mut report = None;
        let mut shapes = Vec::new();
        let mut second_report = None;
        let mut exports = Vec::new();
        // Each export reads its attachments from the vault the previous one
        // wrote — the only root the graph it is exporting was ever built from.
        let mut current = Arc::new(graph.clone());
        for (round, dir) in dirs.iter().enumerate() {
            let done = export(
                &current,
                dir.path(),
                &ExportOptions {
                    source_root: root.take(),
                    ..ExportOptions::default()
                },
            )
            .unwrap();
            if let Some(config) = &config {
                let into = crate::okf::vault_config::config_path(dir.path());
                std::fs::create_dir_all(into.parent().unwrap()).unwrap();
                std::fs::copy(config, into).unwrap();
            }
            exports.push(snapshot(dir.path()));
            let built = crate::okf::build(dir.path(), &vault_opts()).unwrap();
            if round == 0 {
                report = Some(done);
                second_report = Some(built.report.clone());
            }
            shapes.push(Shape::of(&built.graph));
            root = carry_root.then(|| dir.path().to_path_buf());
            current = built.graph;
        }
        let mut shapes = shapes.into_iter();
        Trip {
            exports,
            first,
            second: shapes.next().unwrap(),
            third: shapes.next().unwrap(),
            report: report.unwrap(),
            second_report: second_report.unwrap(),
            _dirs: dirs,
        }
    }

    /// The properties every vault in the corpus must satisfy: an exported tree
    /// is a fixed point of `export ∘ import`, and so is the graph it builds.
    ///
    /// The comparison starts at the *second* export, because §10.9's losses are
    /// taken on the way out of the author's vault and one of them — a `types:`
    /// declaration §4.2's inference does not agree with — moves the bytes once.
    /// [`Trip::assert_byte_identical_from_the_first_export`] adds the stronger
    /// claim for a vault that takes no such loss.
    fn assert_stable(&self, name: &str) {
        self.assert_trees_match(name, 1, 2);
        assert_eq!(
            self.second, self.third,
            "{name}: the graph drifted on the second round trip, so the losses are not one-shot"
        );
    }

    /// The whole chain is byte-identical, from the very first export.
    fn assert_byte_identical_from_the_first_export(&self, name: &str) {
        self.assert_trees_match(name, 0, 1);
        self.assert_stable(name);
    }

    fn assert_trees_match(&self, name: &str, a: usize, b: usize) {
        let (left, right) = (&self.exports[a], &self.exports[b]);
        assert_eq!(
            left.keys().collect::<Vec<_>>(),
            right.keys().collect::<Vec<_>>(),
            "{name}: export {a} and export {b} wrote different file lists"
        );
        // Content before the manifest: the manifest is one hash per file, so a
        // single changed note reddens it too and a diff of hashes says nothing
        // about what moved.
        let (manifest, content): (Vec<_>, Vec<_>) = left
            .iter()
            .partition(|(rel, _)| rel.ends_with(MANIFEST_FILE));
        for (rel, bytes) in content.into_iter().chain(manifest) {
            assert_eq!(
                String::from_utf8_lossy(bytes),
                String::from_utf8_lossy(&right[rel]),
                "{name}: {rel} differs between export {a} and export {b}"
            );
        }
    }
}

// ── the corpus ─────────────────────────────────────────────────────────────

/// Nested maps, every class of string the quoting rule has to protect, the
/// date inference §4.2 runs at the top level but not inside a sequence, and a
/// `types:` declaration holding one date-like string as text.
const TYPING: &[(&str, &str)] = &[
    (
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\ntypes:\n  Note: {stays_text: string}\n",
    ),
    (
        "notes/typed.md",
        "---\n\
         title: Typed values\n\
         metadata:\n  source: vendor\n  nested:\n    deep: 3\n\
         count: 7\n\
         ratio: 1.5\n\
         whole: 2.0\n\
         flag: true\n\
         released: 2026-01-15\n\
         stamp: 2026-01-15T08:30:00Z\n\
         stays_text: \"2026-02-20\"\n\
         looks_numeric: \"12\"\n\
         looks_hex: \"0x1f\"\n\
         looks_bool: \"yes\"\n\
         looks_null: \"Null\"\n\
         leading_dash: \"- not a sequence\"\n\
         colon_pair: \"key: value\"\n\
         trailing_colon: \"ends:\"\n\
         comment_opener: \"text # not a comment\"\n\
         padded: \"  padded  \"\n\
         empty: \"\"\n\
         brace: \"{not a map}\"\n\
         at_sign: \"@handle\"\n\
         tags: [alpha, 2026-01-15]\n\
         ---\n\
         Prose about typed values.\n",
    ),
];

/// Both folder-note spellings, under a folder that *is* the label so §10.2
/// preserves them; `parent:` extras; and two notes sharing a stem, which forces
/// the folder-qualified `[[Label/Stem]]` spelling on every link to them.
const LAYOUT: &[(&str, &str)] = &[
    (
        "Docs/Guides.md",
        "---\ntitle: Guides\n---\nThe folder note for `Docs/Guides/`, the `X.md` spelling.\n",
    ),
    (
        "Docs/Survey/Survey.md",
        "---\ntitle: Survey\n---\nThe folder note for `Docs/Survey/`, the `X/X.md` spelling.\n",
    ),
    (
        "Docs/Guides/Intro.md",
        "---\nparent: \"[[Survey]]\"\ndepends_on: \"[[Docs/Guides/Faults]]\"\n---\n\
         Cross-listed under Survey as well as its own folder.\n",
    ),
    (
        "Docs/Guides/Faults.md",
        "Its stem collides with `Docs/Survey/Faults.md`, so both fall back to their paths.\n",
    ),
    (
        "Docs/Survey/Faults.md",
        "The other Faults. A link to either has to name the folder.\n",
    ),
];

/// Attachments: an image reached note-relative, one reached on the bare
/// filename rung, a non-image attachment, and one that is simply not there.
const MEDIA: &[(&str, &str)] = &[
    (
        "Media/figures.md",
        "---\ntitle: Figures\n---\n\
         ## Plates\n\n\
         ![Fault map](../img/faults.png) is note-relative, ![[diagram.png]] resolves\n\
         on the bare filename, and ![[handbook.pdf]] is not an image at all.\n\n\
         ## Gaps\n\n\
         ![missing](../img/absent.png) resolves to nothing.\n",
    ),
    ("img/faults.png", "stand-in bytes\n"),
    ("img/diagram.png", "other stand-in bytes\n"),
    ("img/handbook.pdf", "not a pdf either\n"),
];

/// A declared edge table (VAULT.md §7.3, §10.6): the `structure.tables … edges:
/// true` rule that reads one in, and the `export.edge_tables:` entry that
/// writes it back. The shape is `vault-structure/structure/tables.md`'s — a
/// row whose cell carries display text, and one naming a note nobody wrote.
const EDGE_TABLES: &[(&str, &str)] = &[
    (
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\n\
         structure:\n  tables:\n    \
         - {under_heading: '^Worked on by$', edge: WORKED_ON_BY, edges: true}\n\
         export:\n  edge_tables:\n    WORKED_ON_BY: Worked on by\n",
    ),
    (
        "Note/paper.md",
        "# Paper\n\nWho worked on this, and when.\n\n\
         ## Worked on by\n\n\
         | person | role | since |\n|---|---|---|\n\
         | [[chunky\\|The chunky note]] | author | 2024 |\n\
         | [[nobody]] | reviewer | 2025 |\n",
    ),
    ("Note/chunky.md", "The chunky note's own prose.\n"),
];

/// Every edge of one type as `source -> target {props}`, sorted — two
/// independently built graphs compare by value.
fn edges_of(graph: &DirGraph, conn_type: &str) -> Vec<String> {
    let name = |idx| {
        graph
            .node_view(idx)
            .map(|view| crate::datatypes::values::raw_string(&view.id()))
            .unwrap_or_default()
    };
    let mut out = Vec::new();
    for edge in graph.graph.edge_indices() {
        let Some(data) = graph.graph.edge_weight(edge) else {
            continue;
        };
        if data.connection_type_str(&graph.interner) != conn_type {
            continue;
        }
        let Some((src, tgt)) = graph.graph.edge_endpoints(edge) else {
            continue;
        };
        let props: BTreeMap<String, String> = data
            .property_iter(&graph.interner)
            .map(|(key, value)| (key.to_string(), crate::datatypes::values::raw_string(value)))
            .collect();
        out.push(format!("{} -> {} {props:?}", name(src), name(tgt)));
    }
    out.sort();
    out
}

/// The phase's claim, end to end: a declared type's edges come back with the
/// properties they left with — `row`, `section`, the cell's display text and
/// every other column — where the author's own `vault.yaml` travelled with the
/// vault (§10.6).
#[test]
fn a_declared_edge_table_carries_its_properties_through_the_round_trip() {
    let dir = vault_of(EDGE_TABLES);
    let source = build_vault(dir.path());
    let before = edges_of(&source, "WORKED_ON_BY");
    assert_eq!(before.len(), 2, "{before:?}");
    assert!(
        before[0].contains("\"label\": \"The chunky note\"")
            && before[0].contains("\"row\": \"1\""),
        "{before:?}"
    );

    let out = tempfile::tempdir().unwrap();
    let report = export(
        &source,
        out.path(),
        &ExportOptions {
            source_root: Some(dir.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        report.edge_properties_dropped, 3,
        "the cells' own prose links"
    );
    assert_eq!(report.warnings, Vec::<String>::new());

    let into = crate::okf::vault_config::config_path(out.path());
    std::fs::create_dir_all(into.parent().unwrap()).unwrap();
    std::fs::copy(crate::okf::vault_config::config_path(dir.path()), into).unwrap();
    let after = build_vault(out.path());
    assert_eq!(
        edges_of(&after, "WORKED_ON_BY"),
        before,
        "every property survived, and the rows kept their order"
    );
}

// ── the two properties, over the whole corpus ──────────────────────────────

/// 2020-01-01T00:00:00Z, stamped on the corpus's attachments before the trip
/// starts. `mtime` is one of the properties [`Shape`] compares, and it comes
/// from `stat` on the file the round in question read — so with a source that
/// carries today's time, a round whose two exports happen to land in the same
/// second compares equal whether or not the modification time travelled. A
/// backdated source makes the comparison say what it claims to say.
fn backdate_attachments(root: &Path) -> std::time::SystemTime {
    let when = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_577_836_800);
    for rel in ["img/faults.png", "img/diagram.png", "img/handbook.pdf"] {
        let path = root.join(rel);
        if path.is_file() {
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(when)
                .unwrap();
        }
    }
    when
}

/// §10.9 lists six losses and no others, so the modification time of a copied
/// attachment is not one of them: it has to reach the second graph unchanged,
/// and the third.
#[test]
fn a_copied_attachment_keeps_its_modification_time_across_the_rounds() {
    let dir = vault_of(MEDIA);
    backdate_attachments(dir.path());
    let trip = Trip::from_vault(dir.path());
    let source = trip
        .first
        .props(IMAGE_LABEL, "img/faults.png")
        .and_then(|p| p.get("mtime"))
        .cloned()
        .expect("the source image has an mtime");
    assert_eq!(source, "2020-01-01 00:00:00", "the backdating took");
    for (round, shape) in [("second", &trip.second), ("third", &trip.third)] {
        assert_eq!(
            shape
                .props(IMAGE_LABEL, "img/faults.png")
                .and_then(|p| p.get("mtime")),
            Some(&source),
            "the {round} graph's image was stamped with the time of the copy"
        );
    }
}

#[test]
fn every_corpus_vault_reaches_a_fixed_point() {
    // `typing` is the one that declares a `types:` entry §4.2 re-infers
    // differently, so only it takes a loss between the first export and the
    // second; every other vault is byte-identical from the first.
    Trip::from_vault(&golden_vault()).assert_byte_identical_from_the_first_export("golden");
    for (name, files) in [("layout", LAYOUT), ("media", MEDIA)] {
        let dir = vault_of(files);
        Trip::from_vault(dir.path()).assert_byte_identical_from_the_first_export(name);
    }
    let dir = vault_of(TYPING);
    Trip::from_vault(dir.path()).assert_stable("typing");
    // The declared edge table is the other vault whose first export moves
    // bytes: the exporter owns that table and rewrites it whole (§10.6), so
    // the author's own delimiter row is normalised once and never again.
    let dir = vault_of(EDGE_TABLES);
    Trip::carrying_config(dir.path()).assert_stable("edge tables");
    // Without a source root the attachments do not travel, which is a
    // different second vault — and it has to be a fixed point too.
    let dir = vault_of(MEDIA);
    Trip::rootless(dir.path())
        .assert_byte_identical_from_the_first_export("media (no source root)");
}

#[test]
fn a_graph_that_was_never_a_vault_reaches_a_fixed_point() {
    let graph = cypher_graph();
    Trip::from_graph(&graph, None, Shape::of(&graph), None)
        .assert_byte_identical_from_the_first_export("cypher");
}

/// A graph built by `CREATE`, carrying no `file_path` anywhere: every node in
/// it is a note (§10.1), and nothing about it came from a vault — so nothing
/// the export drops can be re-derived from prose it never wrote.
fn cypher_graph() -> DirGraph {
    let mut graph = crate::graph::storage::mode::new_dir_graph_in_mode(
        crate::graph::storage::mode::StorageMode::Memory,
        None,
    )
    .unwrap();
    let params = std::collections::HashMap::new();
    crate::graph::session::execute::execute_mut(
        &mut graph,
        "CREATE (a:Paper {concept_id: 'alpha', title: 'Alpha', body: 'The first paper.', \
         year: 2026, score: 1.5, open: true}), \
         (b:Paper {concept_id: 'beta', title: 'Beta', body: 'The second paper.', year: 2025}), \
         (a)-[:CITES {section: 'Background', anchor: 'intro'}]->(b)",
        &crate::graph::session::execute::ExecuteOptions::eager(&params),
    )
    .unwrap();
    graph
}

// ── §10.9, loss by loss ────────────────────────────────────────────────────

/// Loss 1, as narrowed by this phase: edge properties are not written **for a
/// type `export.edge_tables` does not declare**, and the report says how many
/// went. Where no body states the edge — a graph that was never a vault — they
/// are simply gone.
#[test]
fn an_undeclared_types_properties_are_dropped_where_no_prose_restates_them() {
    let graph = cypher_graph();
    let trip = Trip::from_graph(&graph, None, Shape::of(&graph), None);
    assert_eq!(
        trip.first.edge_props.get("CITES"),
        Some(&2),
        "the created edge carries `section` and `anchor`"
    );
    assert_eq!(
        trip.report.edge_properties_dropped, 2,
        "and the export counts what it dropped"
    );
    assert_eq!(
        trip.second.edge("CITES"),
        1,
        "the edge itself survives as a frontmatter wikilink"
    );
    assert_eq!(
        trip.second.edge_props.get("CITES"),
        Some(&0),
        "but with nothing on it"
    );
}

/// The other half of loss 1: the same graph, one declaration apart. A declared
/// type's properties are written as a table in the source note's own prose, so
/// nothing is dropped and nothing is counted (VAULT.md §7.3, §10.6).
#[test]
fn a_declared_types_properties_are_written_instead_of_dropped() {
    let graph = cypher_graph();
    let dir = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        dir.path(),
        &ExportOptions {
            edge_tables: BTreeMap::from([("CITES".to_string(), "Cites".to_string())]),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.edge_properties_dropped, 0);
    let alpha = std::fs::read_to_string(dir.path().join("Paper/Alpha.md")).unwrap();
    assert!(
        alpha.contains("## Cites\n\n| target |\n| --- |\n| [[Beta#intro]] |\n"),
        "`section` is the heading and `anchor` rides the link: {alpha}"
    );
    assert!(
        !alpha.contains("cites:"),
        "and not a frontmatter key too: {alpha}"
    );
}

/// Loss 1, refined — the claim §10.9 makes flatly is narrower than it reads,
/// so the spec says so. An edge the *body* states keeps its properties, because
/// the body travels verbatim and the next import re-derives them from the same
/// prose with the same scanner.
#[test]
fn a_body_stated_edge_keeps_its_properties() {
    let dir = vault_of(&[
        ("Note/a.md", "## Deep dive\n\n[[b#anchor]]\n"),
        ("Note/b.md", "prose\n"),
    ]);
    let trip = Trip::from_vault(dir.path());
    assert_eq!(trip.report.edge_properties_dropped, 2);
    assert_eq!(
        trip.first.edge_props.get("LINKS_TO"),
        Some(&2),
        "`section` and `anchor`"
    );
    assert_eq!(
        trip.second.edge_props.get("LINKS_TO"),
        Some(&2),
        "the prose said it, so the re-import says it again"
    );
    assert_eq!(trip.second.edge("LINKS_TO"), 1, "and does not double it");
}

/// Loss 2. Attachment bytes travel only when the caller names the root they
/// were read from; without it the references are counted unresolvable and come
/// back as `missing: true` stubs.
#[test]
fn attachment_bytes_travel_only_with_a_source_root() {
    let dir = vault_of(MEDIA);
    let carried = Trip::from_vault(dir.path());
    assert_eq!(
        (
            carried.report.attachments_copied,
            carried.report.attachments_unresolved
        ),
        (3, 0)
    );
    assert_eq!(carried.first.label("Image"), 3, "two real, one missing");
    assert_eq!(carried.second.label("Image"), 3);
    assert_eq!(carried.second_report.missing_attachments, 1);

    let rootless = Trip::rootless(dir.path());
    assert_eq!(
        (
            rootless.report.attachments_copied,
            rootless.report.attachments_unresolved
        ),
        (0, 3)
    );
    assert_eq!(
        rootless.second_report.missing_attachments, 4,
        "every reference now names a file the exported vault does not hold"
    );
    assert_eq!(
        rootless
            .second
            .props(ATTACHMENT_LABEL, "handbook.pdf")
            .and_then(|p| p.get("missing"))
            .map(String::as_str),
        Some("true"),
        "the pdf did not travel, so nothing about it is known but its absence —          §6.6 still labels the stub from the extension"
    );
    assert_eq!(
        rootless
            .second
            .props(ATTACHMENT_LABEL, "handbook.pdf")
            .unwrap()
            .get("size_bytes"),
        None,
        "and `stat` said nothing, because there was nothing to stat"
    );
}

/// Loss 3. Synthesized nodes are not files: hubs, folders, tags, attachments
/// and stubs are whatever the *new* layout regenerates, not what the old one
/// held. The golden vault's `Keyword` hub needs `.kglite/vault.yaml`, so it
/// does not come back at all; its `Folder` nodes follow the new one-folder-per
/// -label layout; its `Tag`, `Image` and stub nodes are re-made from prose.
#[test]
fn synthesized_nodes_are_regenerated_not_carried() {
    let trip = Trip::from_vault(&golden_vault());
    assert!(trip.first.label("Keyword") > 0);
    assert_eq!(
        trip.second.label("Keyword"),
        0,
        "a hub needs its declaration"
    );
    assert_eq!(trip.second.edge("HAS_KEYWORD"), 0);
    assert_ne!(
        trip.first.label(FOLDER_LABEL),
        trip.second.label(FOLDER_LABEL),
        "the folders are the export's layout, not the author's"
    );
    for regenerated in [TAG_LABEL, IMAGE_LABEL, "Concept"] {
        assert_eq!(
            trip.first.label(regenerated),
            trip.second.label(regenerated),
            "{regenerated} is re-made from the prose, which travelled verbatim"
        );
    }
}

/// Loss 4. `.kglite/vault.yaml` is not written, so everything it declared is
/// re-derived from the defaults: hubs (above), index and text-index
/// declarations, the `embed:` targets, and `heading_edges` retyping — which
/// shows up as the built-in ladder's own type appearing *beside* the declared
/// one, because the exporter keeps the declared edge and the prose still says
/// what the ladder reads.
#[test]
fn the_vault_yaml_declarations_are_not_carried() {
    let first_report = crate::okf::build(&golden_vault(), &vault_opts())
        .unwrap()
        .report;
    assert!(first_report.indexes_declared > 0);
    assert!(first_report.text_indexes_built > 0);
    assert!(!first_report.embed_targets.is_empty());

    let trip = Trip::from_vault(&golden_vault());
    assert_eq!(trip.second_report.indexes_declared, 0);
    assert_eq!(trip.second_report.text_indexes_built, 0);
    assert!(trip.second_report.embed_targets.is_empty());

    assert_eq!(trip.first.edge("RELATED"), 0, "the declaration retyped it");
    assert!(trip.first.edge("RELATED_TO") > 0);
    assert_eq!(
        trip.second.edge("RELATED_TO"),
        trip.first.edge("RELATED_TO"),
        "the typed edge itself is written as a frontmatter key and survives"
    );
    assert!(
        trip.second.edge("RELATED") > 0,
        "but the same prose now also reads as the built-in ladder's RELATED"
    );
}

/// Loss 5. A top-level string that looks like a date comes back a date. Only
/// `types:` stops §4.2's inference, quoting does not, and `types:` lives in the
/// file the export does not write.
#[test]
fn a_declared_string_that_looks_like_a_date_comes_back_a_date() {
    let dir = vault_of(TYPING);
    let trip = Trip::from_vault(dir.path());
    assert_eq!(
        trip.first
            .props("Note", "typed")
            .and_then(|p| p.get("stays_text"))
            .map(String::as_str),
        Some("2026-02-20"),
        "the declaration kept it text"
    );
    let second = trip
        .second
        .props("Note", "typed")
        .expect("the note came back");
    assert_eq!(
        second.get("stays_text").map(String::as_str),
        Some("2026-02-20")
    );
    // The rendered text is the same; the *type* is not, which is what the next
    // `WHERE n.stays_text < date('…')` sees. `DateTime` is the engine's name
    // for a date-without-time; `Timestamp` is the one with a clock on it.
    assert_eq!(
        property_type(dir.path(), "stays_text"),
        "String",
        "before the round trip the declaration held"
    );
    assert_eq!(
        property_type(trip.exported_dir(), "stays_text"),
        "DateTime",
        "after it — §4.2 infers, and nothing is left to say otherwise"
    );
}

/// Loss 5, refined — the same sentence in §10.9 used to reach every declared
/// type. It does not: the writer emits an `int` as an int and §4.2 reads it
/// back as one, so a declaration that merely *agreed* with what the emitted
/// spelling infers survives without the file that made it.
#[test]
fn a_declared_int_survives_the_round_trip() {
    let dir = vault_of(&[
        (
            ".kglite/vault.yaml",
            "kglite_vault: 1\ndefault_label: Note\ntypes:\n  Note: {toc_depth: int}\n",
        ),
        ("notes/atlas.md", "---\ntoc_depth: \"2\"\n---\nprose\n"),
    ]);
    assert_eq!(property_type(dir.path(), "toc_depth"), "Int64");
    let trip = Trip::from_vault(dir.path());
    assert_eq!(
        property_type(trip.exported_dir(), "toc_depth"),
        "Int64",
        "the export wrote an int, and inference agrees with the declaration"
    );
    assert_eq!(
        trip.second
            .props("Note", "atlas")
            .and_then(|p| p.get("toc_depth"))
            .map(String::as_str),
        Some("2")
    );
}

/// **Not** a loss: a re-filed note keeps the stem its links name.
///
/// The stem is the link namespace (§3, §5.2) and the title is display, so an
/// export that cannot preserve a note's folder still preserves its filename.
/// Naming the re-filed file after the *title* instead dangles every body
/// wikilink that spelled the old stem, and the next import mints a
/// `_provisional` stub for each — 110 of them on the Petrel round trip.
#[test]
fn a_re_filed_note_keeps_the_stem_its_links_name() {
    // Both notes sit at the vault root, which no label can match, so §10.2's
    // preservation rung cannot keep either path.
    let dir = vault_of(&[
        (
            "AB.md",
            "---\ntitle: A/B\n---\nThe note the other one names.\n",
        ),
        ("Other.md", "See [[AB]].\n"),
    ]);
    let trip = Trip::from_vault(dir.path());
    assert_eq!(trip.first.label("Note"), 2);
    assert_eq!(trip.first.edge("LINKS_TO"), 1);
    assert!(
        trip.exports[0].contains_key("Note/AB.md"),
        "only the folder moves: {:?}",
        trip.exports[0].keys().collect::<Vec<_>>()
    );
    assert_eq!(
        trip.second.label("Note"),
        2,
        "the body's `[[AB]]` still resolves, so no stub is minted"
    );
    assert_eq!(trip.second.label("Concept"), 0);
    assert_eq!(trip.second.edge("LINKS_TO"), 1);
    // And the title the stem does not spell is written out, so it survives.
    assert!(
        String::from_utf8_lossy(&trip.exports[0]["Note/AB.md"]).contains("title: A/B\n"),
        "{}",
        String::from_utf8_lossy(&trip.exports[0]["Note/AB.md"])
    );
    trip.assert_byte_identical_from_the_first_export("re-filed stem");
}

/// Loss 6. A re-filed note carries its body verbatim, so a note-relative
/// reference in that body resolves from where the note *now* is.
#[test]
fn a_re_filed_notes_relative_reference_moves_with_it() {
    let dir = vault_of(&[
        (
            "deep/nested/note.md",
            "---\ntype: Media\ntitle: Nested\n---\n![chart](../../img/chart.png)\n",
        ),
        ("img/chart.png", "stand-in bytes\n"),
    ]);
    let build = crate::okf::build(dir.path(), &vault_opts()).unwrap();
    assert!(build.report.errors.is_empty(), "{:?}", build.report.errors);
    let trip = Trip::from_vault(dir.path());
    assert!(
        trip.first.props(IMAGE_LABEL, "img/chart.png").is_some(),
        "it resolved from `deep/nested/`"
    );
    // `type: Media` and a top folder of `deep` do not match, so the note is
    // re-filed to `Media/note.md` — keeping its stem, but one level up, where
    // `../../` climbs out of the vault entirely.
    assert!(trip.exports[0].contains_key("Media/note.md"));
    assert!(
        trip.second_report
            .errors
            .iter()
            .any(|e| e.contains("escapes the vault root")),
        "the same two dots now name a place outside the vault: {:?}",
        trip.second_report.errors
    );
    // §6.2's third rung — a unique filename anywhere in the vault — still finds
    // the copied file, so the picture is not lost; the *reference* is wrong, and
    // a vault holding two `chart.png`s would resolve to neither.
    assert!(trip.second.props(IMAGE_LABEL, "img/chart.png").is_some());
}

/// A typed inline link (VAULT.md §5.3 rung 0) is a statement the prose already
/// carries, so the export must leave the body alone and must not restate the
/// edge as a frontmatter `see_also:` key — which would make a second edge on
/// the next import.
#[test]
fn a_typed_inline_link_round_trips_in_the_prose_alone() {
    let body = "## Related work\n\nRead [[b|the other one]]{see-also}.\n";
    let dir = vault_of(&[("Note/a.md", body), ("Note/b.md", "prose\n")]);
    let trip = Trip::from_vault(dir.path());
    assert_eq!(trip.first.edge("SEE_ALSO"), 1);
    assert_eq!(trip.first.edge("LINKS_TO"), 0);
    let written = String::from_utf8(trip.exports[0]["Note/a.md"].clone()).unwrap();
    assert!(
        written.ends_with(body),
        "the body is written verbatim, brace and all: {written:?}"
    );
    assert!(
        !written.contains("see_also"),
        "the prose states the edge, so the frontmatter must not: {written:?}"
    );
    assert_eq!(
        (trip.second.edge("SEE_ALSO"), trip.second.edge("LINKS_TO")),
        (1, 0),
        "and the re-import reads exactly the one edge back"
    );
}

/// The declared type of one property, read out of a freshly built vault — the
/// thing a rendered value cannot tell you (`"2026-02-20"` and a `Date` print
/// the same).
fn property_type(vault: &Path, property: &str) -> String {
    let graph = build_vault(vault);
    for idx in graph.graph.node_indices() {
        let Some(view) = graph.node_view(idx) else {
            continue;
        };
        if let Some(value) = view.get_property_value(property) {
            return value.type_name().to_string();
        }
    }
    panic!("no node carries `{property}` in {}", vault.display());
}

impl Trip {
    /// The directory the first export wrote into — kept alive by `_dirs`.
    fn exported_dir(&self) -> &Path {
        self._dirs[0].path()
    }
}

/// A `<!-- kglite … -->` directive is metadata to the *graph* and prose to the
/// *file*: nothing derived carries it, and the body property still holds it
/// byte for byte, so it travels an export unchanged (VAULT.md §5.8, §10.9).
#[test]
fn a_directive_survives_the_round_trip_byte_for_byte() {
    let files: &[(&str, &str)] = &[
        (
            ".kglite/vault.yaml",
            "kglite_vault: 1\ndefault_label: Note\nstructure:\n  \
             sections: {label: Section, edge: HAS_SECTION, parent: PARENT_SECTION, next: NEXT_SECTION}\n  \
             chunks: {label: Chunk, edge: HAS_CHUNK, next: NEXT_CHUNK, max_words: 650, max_chars: 6000}\n",
        ),
        (
            "Annotations.md",
            "---\ntitle: Annotations\n---\n### Annotation Table\n\n\
             To open the **Annotation Table** dialog box, click the button.\n\n\
             <!-- kglite address: Data tree -> Wells | Task pane -->\n\n\
             Then pick a well.\n",
        ),
    ];
    let dir = vault_of(files);
    let trip = Trip::carrying_config(dir.path());
    trip.assert_byte_identical_from_the_first_export("directives");
    let exported = String::from_utf8(trip.exports[0]["Note/Annotations.md"].clone()).unwrap();
    assert!(
        exported.contains("<!-- kglite address: Data tree -> Wells | Task pane -->"),
        "the export wrote the author's own directive line back: {exported}"
    );
    // And the graph read from it never carried the directive as prose.
    let chunk = trip
        .second
        .props("Chunk", "Annotations#Annotation Table~chunk1")
        .expect("the chunk the two paragraphs packed into");
    assert!(
        !chunk["text"].contains("kglite"),
        "the chunk is prose only: {:?}",
        chunk["text"]
    );
}
