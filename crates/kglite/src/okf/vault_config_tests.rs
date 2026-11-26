//! `.kglite/vault.yaml` (VAULT.md §7) and the skills and recipes a vault
//! carries (§8).
//!
//! The schema tests go through [`parse`] (no directory needed); everything
//! that has to *reach* a graph goes through a real [`crate::okf::build`] over
//! a temp vault, because a config field nothing reads is exactly the failure
//! these tests exist to catch.

use super::*;
use crate::graph::storage::GraphRead;
use crate::okf::model::BuildOptions;
use crate::okf::{build::BuildOutput, Dialect};
use std::fs;
use tempfile::{tempdir, TempDir};

/// A vault directory with `body` in every note and the given `vault.yaml`.
fn vault_with(config: Option<&str>, notes: &[(&str, &str)]) -> TempDir {
    let dir = tempdir().unwrap();
    if let Some(text) = config {
        fs::create_dir_all(dir.path().join(CONFIG_DIR)).unwrap();
        fs::write(config_path(dir.path()), text).unwrap();
    }
    for (rel, content) in notes {
        let path = dir.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }
    dir
}

fn build_vault(dir: &TempDir) -> BuildOutput {
    build_as(dir, Dialect::Obsidian).expect("the vault built")
}

fn build_as(dir: &TempDir, dialect: Dialect) -> Result<BuildOutput, String> {
    crate::okf::build(dir.path(), &BuildOptions::for_dialect(dialect))
}

/// Every `(label, id)` in the graph, sorted — the cheapest way to say what a
/// profile override did to the notes.
fn labels(out: &BuildOutput) -> Vec<(String, String)> {
    let graph = &out.graph;
    let mut rows: Vec<(String, String)> = graph
        .graph
        .node_indices()
        .filter_map(|n| {
            let view = graph.node_view(n)?;
            Some((
                view.node_type_str(&graph.interner).to_string(),
                crate::datatypes::values::raw_string(&view.id()),
            ))
        })
        .collect();
    rows.sort();
    rows
}

fn property(out: &BuildOutput, label: &str, id: &str, key: &str) -> Option<Value> {
    let graph = &out.graph;
    graph.graph.node_indices().find_map(|n| {
        let view = graph.node_view(n)?;
        (view.node_type_str(&graph.interner) == label
            && crate::datatypes::values::raw_string(&view.id()) == id)
            .then(|| view.get_property_value(key))?
    })
}

/// The `(source id, target id)` of every edge of one type, sorted.
fn edge_endpoints(out: &BuildOutput, conn_type: &str) -> Vec<(String, String)> {
    let graph = &out.graph;
    let id_of = |n| {
        graph
            .node_view(n)
            .map(|v| crate::datatypes::values::raw_string(&v.id()))
    };
    let mut rows: Vec<(String, String)> = graph
        .graph
        .edge_indices()
        .filter_map(|e| {
            let (src, tgt) = graph.graph.edge_endpoints(e)?;
            (graph
                .graph
                .edge_weight(e)?
                .connection_type_str(&graph.interner)
                == conn_type)
                .then(|| Some((id_of(src)?, id_of(tgt)?)))?
        })
        .collect();
    rows.sort();
    rows
}

/// A node's display title, which lives in its own field rather than in the
/// property bag `property` reads.
fn title(out: &BuildOutput, label: &str, id: &str) -> Option<String> {
    let graph = &out.graph;
    graph.graph.node_indices().find_map(|n| {
        let view = graph.node_view(n)?;
        (view.node_type_str(&graph.interner) == label
            && crate::datatypes::values::raw_string(&view.id()) == id)
            .then(|| crate::datatypes::values::raw_string(&view.title()))
    })
}

const MINIMAL: &str = "kglite_vault: 1\n";

// ── Loading ────────────────────────────────────────────────────────────────

#[test]
fn a_vault_without_a_config_file_loads_none() {
    let dir = vault_with(None, &[("a.md", "prose")]);
    assert_eq!(load(dir.path()), Ok(None));
    // …and the build is the bare dialect's.
    assert!(build_vault(&dir).report.warnings.is_empty());
}

#[test]
fn the_version_key_is_required_and_closed() {
    assert!(parse("default_label: Article\n")
        .unwrap_err()
        .contains("`kglite_vault: 1` is required"));
    let wrong = parse("kglite_vault: 2\n").unwrap_err();
    assert!(
        wrong.contains("kglite_vault: 2") && wrong.contains("supported version is 1"),
        "{wrong}"
    );
    assert!(parse("kglite_vault: one\n")
        .unwrap_err()
        .contains("must be the integer 1"));
    assert!(parse("").unwrap_err().contains("required"));
    assert!(parse("- a\n- b\n").unwrap_err().contains("YAML mapping"));
    assert_eq!(parse(MINIMAL), Ok(VaultConfig::default()));
}

#[test]
fn an_unknown_top_level_key_is_an_error() {
    let err = parse("kglite_vault: 1\nheading_edge: {a: B}\n").unwrap_err();
    assert!(
        err.contains("unknown key `heading_edge`") && err.contains("heading_edges"),
        "the message names the typo and the accepted set: {err}"
    );
}

#[test]
fn a_broken_config_fails_the_build_rather_than_being_ignored() {
    let dir = vault_with(Some("kglite_vault: 9\n"), &[("a.md", "prose")]);
    let err = build_as(&dir, Dialect::Obsidian)
        .err()
        .expect("the build failed");
    assert!(
        err.contains("vault.yaml") && err.contains("supported version is 1"),
        "the path and the rule: {err}"
    );
}

#[test]
fn okf_and_loose_ignore_the_file_with_a_warning() {
    let dir = vault_with(
        Some("kglite_vault: 1\ndefault_label: Article\n"),
        &[("a.md", "---\ntitle: A\n---\nprose")],
    );
    for (dialect, name) in [(Dialect::Okf, "okf"), (Dialect::Loose, "loose")] {
        let out = build_as(&dir, dialect).unwrap();
        assert_eq!(out.report.warnings.len(), 1, "{:?}", out.report.warnings);
        assert!(
            out.report.warnings[0].contains(".kglite/vault.yaml")
                && out.report.warnings[0].contains(name),
            "{}",
            out.report.warnings[0]
        );
        assert!(
            labels(&out).iter().all(|(label, _)| label != "Article"),
            "the declared label was not applied"
        );
    }
    // A bundle with no config warns about nothing.
    let bare = vault_with(None, &[("a.md", "---\ntitle: A\n---\nprose")]);
    assert!(build_as(&bare, Dialect::Okf)
        .unwrap()
        .report
        .warnings
        .is_empty());
}

#[test]
fn the_spec_example_parses_and_is_read_whole() {
    // VAULT.md §7's complete example, verbatim. It is the document the spec
    // shows a converter author, so it is the one that has to load.
    let config = parse(
        r#"
kglite_vault: 1
default_label: Article
body: body

folder_notes:
  edge: CHILD_OF
  direction: child_to_parent

hubs:
  keywords: {label: Keyword, edge: HAS_KEYWORD, case_insensitive: true}
  component: {label: Component, edge: USES_COMPONENT, case_insensitive: true}

heading_edges:
  "Related topics": RELATED_TO

types:
  Article: {description: string, toc_depth: int, updated: date}

indexes:
  Article:
    - concept_id
    - title
    - {range: toc_depth}
  Keyword: [concept_id]
  Component: [concept_id]

text_indexes:
  Article: [body]

embed:
  Article: description
"#,
    )
    .expect("the spec's example is valid");

    assert_eq!(config.default_label.as_deref(), Some("Article"));
    assert_eq!(config.body.as_deref(), Some("body"));
    assert_eq!(config.folder_note_edge.as_deref(), Some("CHILD_OF"));
    assert_eq!(
        config.folder_note_direction,
        Some(FolderNoteDirection::ChildToParent)
    );
    assert_eq!(config.hubs.len(), 2);
    assert_eq!(
        config.hubs.get("keywords"),
        Some(&HubSpec {
            label: "Keyword".to_string(),
            edge: "HAS_KEYWORD".to_string(),
            case_insensitive: true,
        })
    );
    assert_eq!(
        config
            .heading_edges
            .get("Related topics")
            .map(String::as_str),
        Some("RELATED_TO")
    );
    assert_eq!(
        config
            .types
            .get("Article")
            .and_then(|t| t.get("toc_depth"))
            .map(String::as_str),
        Some("int")
    );
    assert_eq!(
        config.indexes.get("Article"),
        Some(&vec![
            IndexDecl::Equality("concept_id".to_string()),
            IndexDecl::Equality("title".to_string()),
            IndexDecl::Range("toc_depth".to_string()),
        ])
    );
    assert_eq!(
        config.text_indexes.get("Article"),
        Some(&vec!["body".to_string()])
    );
    assert_eq!(
        config.embed,
        vec![("Article".to_string(), "description".to_string())]
    );
}

// ── Profile overrides ──────────────────────────────────────────────────────

#[test]
fn default_label_and_label_from_reach_the_notes() {
    let notes = [("guides/a.md", "prose"), ("b.md", "prose")];
    let dir = vault_with(Some("kglite_vault: 1\ndefault_label: Article\n"), &notes);
    assert_eq!(
        labels(&build_vault(&dir)),
        vec![
            ("Article".to_string(), "a".to_string()),
            ("Article".to_string(), "b".to_string()),
            ("Folder".to_string(), "guides".to_string()),
        ],
        "`default_label` sits ahead of the folder rung"
    );

    let folder_first = vault_with(
        Some("kglite_vault: 1\ndefault_label: Article\nlabel_from: folder\n"),
        &notes,
    );
    assert_eq!(
        labels(&build_vault(&folder_first)),
        vec![
            ("Article".to_string(), "b".to_string()),
            ("Folder".to_string(), "guides".to_string()),
            ("guides".to_string(), "a".to_string()),
        ],
        "`label_from: folder` moves the folder rung in front of `default_label`"
    );
}

#[test]
fn body_renames_the_prose_property() {
    let dir = vault_with(
        Some("kglite_vault: 1\nbody: prose\n"),
        &[(
            "a.md",
            "---\nbody: a frontmatter key of the same name\n---\nThe prose.",
        )],
    );
    let out = build_vault(&dir);
    assert_eq!(
        property(&out, "Note", "a", "prose"),
        Some(Value::String("The prose.".to_string()))
    );
    assert_eq!(
        property(&out, "Note", "a", "body"),
        Some(Value::String(
            "a frontmatter key of the same name".to_string()
        )),
        "the frontmatter key keeps `body`, which is why the rename exists"
    );
}

#[test]
fn skip_dirs_prunes_from_the_config() {
    let dir = vault_with(
        Some("kglite_vault: 1\nskip_dirs: [drafts]\n"),
        &[("keep.md", "prose"), ("drafts/no.md", "prose")],
    );
    assert_eq!(
        labels(&build_vault(&dir)),
        vec![("Note".to_string(), "keep".to_string())],
        "no note and no Folder from `drafts/`"
    );
}

#[test]
fn folder_notes_edge_and_direction_are_declared() {
    let notes = [
        ("projects.md", "---\ntitle: Projects\n---\nthe folder note"),
        ("projects/a.md", "prose"),
    ];
    let dir = vault_with(
        Some("kglite_vault: 1\nfolder_notes: {edge: PART_OF, direction: parent_to_child}\n"),
        &notes,
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.edges_by_type.get("PART_OF"), Some(&1));
    assert_eq!(out.report.edges_by_type.get("CHILD_OF"), None);
    // The direction is which way the edge runs, not how many there are — the
    // folder note is the source under `parent_to_child`.
    assert_eq!(
        edge_endpoints(&out, "PART_OF"),
        vec![("projects".to_string(), "a".to_string())]
    );

    let default_direction = vault_with(
        Some("kglite_vault: 1\nfolder_notes: {edge: PART_OF}\n"),
        &notes,
    );
    assert_eq!(
        edge_endpoints(&build_vault(&default_direction), "PART_OF"),
        vec![("a".to_string(), "projects".to_string())],
        "`child_to_parent` is the default"
    );

    let err = parse("kglite_vault: 1\nfolder_notes: {direction: sideways}\n").unwrap_err();
    assert!(err.contains("child_to_parent"), "{err}");
    let unknown = parse("kglite_vault: 1\nfolder_notes: {edges: X}\n").unwrap_err();
    assert!(unknown.contains("`folder_notes.edges`"), "{unknown}");
}

#[test]
fn a_declared_hub_folds_casing_and_keeps_the_built_in_tag_hub() {
    let dir = vault_with(
        Some(
            "kglite_vault: 1\nhubs:\n  \
             keywords: {label: Keyword, edge: HAS_KEYWORD, case_insensitive: true}\n",
        ),
        &[(
            "a.md",
            "---\nkeywords:\n  - Faults\n  - faults\n  - faults\ntags:\n  - Seismic\n---\nprose",
        )],
    );
    let out = build_vault(&dir);
    assert_eq!(
        title(&out, "Keyword", "faults").as_deref(),
        Some("faults"),
        "the casing the vault used most often"
    );
    assert_eq!(out.report.edges_by_type.get("HAS_KEYWORD"), Some(&1));
    assert_eq!(
        out.report.nodes_by_label.get("Tag"),
        Some(&1),
        "declaring a hub adds to the built-in `tags` one, it does not replace it"
    );

    // …and `tags` itself can be redeclared, naming only what changes: the
    // built-in's own folding survives a redeclaration that is silent about it.
    let relabelled = vault_with(
        Some("kglite_vault: 1\nhubs: {tags: {label: Topic}}\n"),
        &[("a.md", "---\ntags: [Seismic, seismic]\n---\nprose")],
    );
    let out = build_vault(&relabelled);
    assert_eq!(out.report.nodes_by_label.get("Topic"), Some(&1));
    assert_eq!(
        title(&out, "Topic", "seismic").as_deref(),
        Some("Seismic"),
        "a tie in frequency settles alphabetically"
    );

    // A vault that wants the two spellings apart says so, and only `tags`
    // starts folded — a fresh key does not inherit anything.
    let split = vault_with(
        Some(
            "kglite_vault: 1\nhubs:\n  tags: {case_insensitive: false}\n  \
             keywords: {label: Keyword, edge: HAS_KEYWORD}\n",
        ),
        &[(
            "a.md",
            "---\ntags: [Seismic, seismic]\nkeywords: [Faults, faults]\n---\nprose",
        )],
    );
    let out = build_vault(&split);
    assert_eq!(out.report.nodes_by_label.get("Tag"), Some(&2));
    assert_eq!(
        out.report.nodes_by_label.get("Keyword"),
        Some(&2),
        "`case_insensitive` defaults to false for a key naming no built-in hub"
    );
}

#[test]
fn heading_edges_retype_the_links_below_a_heading() {
    let dir = vault_with(
        Some("kglite_vault: 1\nheading_edges: {\"Related topics\": RELATED_TO}\n"),
        &[("a.md", "## Related topics\n\n[[b]]"), ("b.md", "prose")],
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.edges_by_type.get("RELATED_TO"), Some(&1));
    assert_eq!(
        out.report.edges_by_type.get("RELATED"),
        None,
        "the declared map wins over the built-in ladder"
    );
}

#[test]
fn the_config_wins_over_a_caller_set_profile() {
    let dir = vault_with(
        Some("kglite_vault: 1\ndefault_label: Article\n"),
        &[("a.md", "prose")],
    );
    let mut opts = BuildOptions::for_dialect(Dialect::Obsidian);
    opts.profile.default_label = Some("CallerSaidSo".to_string());
    let out = crate::okf::build(dir.path(), &opts).unwrap();
    assert_eq!(
        labels(&out),
        vec![("Article".to_string(), "a".to_string())],
        "a rebuild re-reads the file, so the file has to be the authority"
    );
}

// ── types ──────────────────────────────────────────────────────────────────

#[test]
fn declared_types_override_inference() {
    let dir = vault_with(
        Some(
            "kglite_vault: 1\ntypes:\n  Note:\n    depth: int\n    ratio: float\n    \
             live: bool\n    updated: string\n    when: datetime\n    labels: list\n",
        ),
        &[(
            "a.md",
            "---\ndepth: \"2\"\nratio: \"0.5\"\nlive: \"yes\"\nupdated: 2026-01-15\n\
             when: 2026-01-15\nlabels: [x]\n---\nprose",
        )],
    );
    let out = build_vault(&dir);
    assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);
    assert_eq!(property(&out, "Note", "a", "depth"), Some(Value::Int64(2)));
    assert_eq!(
        property(&out, "Note", "a", "ratio"),
        Some(Value::Float64(0.5))
    );
    assert_eq!(
        property(&out, "Note", "a", "live"),
        Some(Value::Boolean(true))
    );
    assert_eq!(
        property(&out, "Note", "a", "updated"),
        Some(Value::String("2026-01-15".to_string())),
        "`string` is how a vault turns the ISO-date inference off"
    );
    assert_eq!(
        property(&out, "Note", "a", "when"),
        Some(Value::Timestamp(
            chrono::NaiveDate::from_ymd_opt(2026, 1, 15)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
        ))
    );
}

#[test]
fn a_value_that_will_not_coerce_warns_and_is_left_as_written() {
    let dir = vault_with(
        Some("kglite_vault: 1\ntypes: {Note: {depth: int}}\n"),
        &[("a.md", "---\ndepth: deep\n---\nprose")],
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.warnings.len(), 1, "{:?}", out.report.warnings);
    let warning = &out.report.warnings[0];
    assert!(
        warning.contains("a.md")
            && warning.contains("Note.depth")
            && warning.contains("int")
            && warning.contains("deep"),
        "the note, the label, the property, the type and the value: {warning}"
    );
    assert_eq!(
        property(&out, "Note", "a", "depth"),
        Some(Value::String("deep".to_string())),
        "the human's value survives"
    );

    // A scalar under `list` is not silently wrapped in a one-element list: a
    // `keywords: seismic` that meant a list is an authoring mistake, and a hub
    // reads a key's list only (VAULT.md §7).
    let scalar = vault_with(
        Some("kglite_vault: 1\ntypes: {Note: {keywords: list}}\n"),
        &[("a.md", "---\nkeywords: seismic\n---\nprose")],
    );
    let out = build_vault(&scalar);
    assert_eq!(out.report.warnings.len(), 1, "{:?}", out.report.warnings);
    assert!(
        out.report.warnings[0].contains("Note.keywords") && out.report.warnings[0].contains("list"),
        "{}",
        out.report.warnings[0]
    );
    assert_eq!(
        property(&out, "Note", "a", "keywords"),
        Some(Value::String("seismic".to_string()))
    );
}

#[test]
fn the_id_column_is_never_retyped() {
    // `concept_id` is the node's identity and the index built on it, so a
    // declaration naming it must not reach the column: coercing these stems
    // to `int` would null every id and unhook every link.
    let dir = vault_with(
        Some("kglite_vault: 1\ntypes: {Note: {concept_id: int}}\n"),
        &[("alpha.md", "see [[beta]]"), ("beta.md", "prose")],
    );
    let out = build_vault(&dir);
    assert_eq!(
        labels(&out),
        vec![
            ("Note".to_string(), "alpha".to_string()),
            ("Note".to_string(), "beta".to_string()),
        ]
    );
    assert_eq!(
        edge_endpoints(&out, "LINKS_TO"),
        vec![("alpha".to_string(), "beta".to_string())]
    );
    assert_eq!(out.report.warnings.len(), 1, "{:?}", out.report.warnings);
    assert!(out.report.warnings[0].contains("id column is not retyped"));
}

#[test]
fn a_declaration_nothing_matches_warns() {
    let dir = vault_with(
        Some(
            "kglite_vault: 1\ntypes:\n  Note: {absent: int, concept_id: string}\n  \
             Ghost: {x: int}\n",
        ),
        &[("a.md", "prose")],
    );
    let warnings = build_vault(&dir).report.warnings;
    assert_eq!(warnings.len(), 3, "{warnings:?}");
    assert!(warnings
        .iter()
        .any(|w| w.contains("types.Note.concept_id") && w.contains("id column is not retyped")));
    assert!(warnings
        .iter()
        .any(|w| w.contains("types.Note.absent") && w.contains("no note carries")));
    assert!(warnings.iter().any(|w| w.contains("types.Ghost.x")));
}

#[test]
fn an_unknown_type_keyword_is_a_config_error() {
    let err = parse("kglite_vault: 1\ntypes: {Note: {depth: integer}}\n").unwrap_err();
    assert!(
        err.contains("types.Note.depth: integer") && err.contains("int"),
        "{err}"
    );
}

// ── indexes, text indexes, ontology, embed ─────────────────────────────────

#[test]
fn each_index_kind_is_installed() {
    let dir = vault_with(
        Some(
            "kglite_vault: 1\nindexes:\n  Note:\n    - title\n    - {range: depth}\n    \
             - {composite: [depth, title]}\n",
        ),
        &[("a.md", "---\ndepth: 2\n---\nprose")],
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.indexes_declared, 3);
    assert!(out.graph.has_index("Note", "title"));
    assert!(out
        .graph
        .range_indices
        .contains_key(&("Note".to_string(), "depth".to_string())));
    assert!(out
        .graph
        .has_composite_index("Note", &["depth".to_string(), "title".to_string()]));
    assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);
}

#[test]
fn an_index_on_a_label_or_property_the_vault_lacks_warns() {
    let dir = vault_with(
        Some("kglite_vault: 1\nindexes:\n  Ghost: [id]\n  Note: [absent]\n"),
        &[("a.md", "prose")],
    );
    let out = build_vault(&dir);
    assert_eq!(
        out.report.indexes_declared, 1,
        "only the Note one was tried"
    );
    assert!(out
        .report
        .warnings
        .iter()
        .any(|w| w.contains("indexes on `Ghost`")));
    assert!(out
        .report
        .warnings
        .iter()
        .any(|w| w.contains("`Note.absent`") && w.contains("indexed no value")));
}

/// A note's id property is `concept_id` and a hub node's is `id` (VAULT.md
/// §7). The P16 usability probe declared `indexes: {Topic: [concept_id]}` for
/// a hub and got "indexed no value" — the index installs, over a property
/// nothing carries. Both halves are pinned, because the warning is the only
/// thing that tells an author they named the wrong one.
#[test]
fn a_hub_is_indexed_on_id_and_a_note_on_concept_id() {
    let dir = vault_with(
        Some(
            "kglite_vault: 1\nhubs:\n  topics: {label: Topic, edge: ON_TOPIC}\n\
             indexes:\n  Topic: [id]\n  Note: [concept_id]\n",
        ),
        &[("a.md", "---\ntopics: [seismic]\n---\nprose")],
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.indexes_declared, 2);
    assert!(out.graph.has_index("Topic", "id"));
    assert!(out.graph.has_index("Note", "concept_id"));
    assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);

    // …and naming the note's id property on a hub indexes nothing.
    let swapped = vault_with(
        Some(
            "kglite_vault: 1\nhubs:\n  topics: {label: Topic, edge: ON_TOPIC}\n\
             indexes:\n  Topic: [concept_id]\n",
        ),
        &[("a.md", "---\ntopics: [seismic]\n---\nprose")],
    );
    let out = build_vault(&swapped);
    assert!(
        out.report
            .warnings
            .iter()
            .any(|w| w.contains("`Topic.concept_id`") && w.contains("indexed no value")),
        "{:?}",
        out.report.warnings
    );
}

#[test]
fn a_malformed_index_entry_is_a_config_error() {
    assert!(parse("kglite_vault: 1\nindexes: {Note: title}\n")
        .unwrap_err()
        .contains("must be a list"));
    assert!(parse("kglite_vault: 1\nindexes: {Note: [{ranged: x}]}\n")
        .unwrap_err()
        .contains("unknown index kind `ranged`"));
    assert!(
        parse("kglite_vault: 1\nindexes: {Note: [{composite: [a]}]}\n")
            .unwrap_err()
            .contains("at least two properties")
    );
    assert!(
        parse("kglite_vault: 1\nindexes: {Note: [{range: a, composite: [a, b]}]}\n")
            .unwrap_err()
            .contains("single-key map")
    );
}

#[test]
fn a_text_index_is_built_and_a_missing_label_warns() {
    let dir = vault_with(
        Some("kglite_vault: 1\ntext_indexes:\n  Note: [body]\n  Ghost: [body]\n"),
        &[("a.md", "seismic interpretation of the atlas survey")],
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.text_indexes_built, 1);
    assert!(crate::graph::text_indexes::has_text_index(
        &out.graph, "Note", "body"
    ));
    assert!(out
        .report
        .warnings
        .iter()
        .any(|w| w.contains("`Ghost.body`") && w.contains("was not built")));
}

#[test]
fn an_ontology_is_installed_from_the_config() {
    let dir = vault_with(
        Some(
            "kglite_vault: 1\nontology:\n  version: 1\n  classes:\n    \
             Work: {abstract: true}\n    Note: {is_a: Work}\n",
        ),
        &[("a.md", "prose")],
    );
    let out = build_vault(&dir);
    assert!(out.report.errors.is_empty(), "{:?}", out.report.errors);
    assert_eq!(out.graph.ontology.classes.len(), 2);
    assert_eq!(
        out.graph.ontology.ancestors("Note"),
        vec!["Work".to_string()]
    );

    // A document the ontology parser refuses is a *config* error, so it fails
    // the build before a half-declared graph exists.
    let broken = vault_with(
        Some("kglite_vault: 1\nontology: {classes: {Note: {parrent: Work}}}\n"),
        &[("a.md", "prose")],
    );
    let err = build_as(&broken, Dialect::Obsidian)
        .err()
        .expect("the build failed");
    assert!(
        err.contains("vault.yaml") && err.contains("ontology"),
        "{err}"
    );
}

#[test]
fn embed_targets_are_reported_not_computed() {
    let dir = vault_with(
        Some("kglite_vault: 1\nembed:\n  Note: body\n  Ghost: body\n"),
        &[("a.md", "prose")],
    );
    let out = build_vault(&dir);
    assert_eq!(
        out.report.embed_targets,
        vec![
            ("Ghost".to_string(), "body".to_string()),
            ("Note".to_string(), "body".to_string())
        ],
        "reported verbatim — core links no embedder"
    );
    assert!(
        out.graph.embeddings.is_empty(),
        "no vectors were computed here"
    );
    assert!(out
        .report
        .warnings
        .iter()
        .any(|w| w.contains("embed: Ghost.body")));
}

// ── Carried skills and recipes (§8) ────────────────────────────────────────

const SKILL: &str = "---\nname: overview\ndescription: How to query this vault.\n---\n\nBody.";
const RECIPE: &str = r#"---
recipe: vault
name: by_title
description: One note by title.
parameters:
  {type: object, properties: {title: {type: string}},
   required: [title], additionalProperties: false}
recipe_description: Reading the vault.
---

```cypher
MATCH (n) WHERE n.title = $title RETURN n.concept_id AS id
```
"#;

fn carrying(skills: &[(&str, &str)], recipes: &[(&str, &str)]) -> TempDir {
    let dir = vault_with(Some(MINIMAL), &[("a.md", "prose")]);
    for (sub, files) in [(SKILLS_DIR, skills), (RECIPES_DIR, recipes)] {
        if files.is_empty() {
            continue;
        }
        let base = dir.path().join(CONFIG_DIR).join(sub);
        fs::create_dir_all(&base).unwrap();
        for (name, text) in files {
            fs::write(base.join(name), text).unwrap();
        }
    }
    dir
}

#[test]
fn skills_and_recipes_under_kglite_reach_the_graph() {
    let dir = carrying(&[("one.md", SKILL)], &[("one.md", RECIPE)]);
    let out = build_vault(&dir);
    assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);
    assert_eq!(out.report.skills_imported, 1);
    assert_eq!(out.report.recipes_imported, 1);

    let skill = crate::graph::skills::get(&out.graph, "overview").unwrap();
    assert_eq!(skill.description, "How to query this vault.");
    assert_eq!(skill.body.trim(), "Body.");

    let recipe = crate::graph::recipes::get(&out.graph, "vault", "by_title").unwrap();
    assert!(recipe
        .cypher
        .starts_with("MATCH (n) WHERE n.title = $title"));
    assert_eq!(recipe.recipe_description, "Reading the vault.");
    assert_eq!(
        recipe.parameters["properties"]["title"]["type"], "string",
        "the schema stayed nested rather than flattening to a dotted key"
    );

    // …and they are ordinary nodes, so they travel with the graph.
    assert_eq!(out.report.nodes_by_label.get("KgliteSkill"), None);
    assert!(out.graph.has_node_type("KgliteSkill"));
    assert!(out.graph.has_node_type("KgliteRecipe"));
}

/// VAULT.md §8: "A file that omits `recipe_description` inherits the group's
/// from a sibling." Resolving that per file as it was read made it depend on
/// filename order — the P16 usability probe wrote the description in one of
/// six siblings and the five sorting before it were skipped for "expected a
/// non-empty group description". The declaring file here sorts **last**.
#[test]
fn a_recipe_inherits_its_group_description_from_any_sibling() {
    let borrower = RECIPE
        .replace("name: by_title", "name: by_id")
        .replace("recipe_description: Reading the vault.\n", "");
    let dir = carrying(
        &[],
        &[
            ("a_borrows.md", borrower.as_str()),
            ("z_declares.md", RECIPE),
        ],
    );
    let out = build_vault(&dir);
    assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);
    assert_eq!(out.report.recipes_imported, 2);
    assert_eq!(
        crate::graph::recipes::get(&out.graph, "vault", "by_id")
            .unwrap()
            .recipe_description,
        "Reading the vault.",
        "inherited from the sibling that declares it, whatever the read order"
    );
}

/// The other half: a group **nothing** describes is still every member's own
/// failure, warned per file and skipped, because §8's posture for carried
/// content is skip-with-warning rather than a failed build.
#[test]
fn a_recipe_group_no_sibling_describes_is_skipped_with_a_warning() {
    let bare = RECIPE.replace("recipe_description: Reading the vault.\n", "");
    let dir = carrying(&[], &[("only.md", bare.as_str())]);
    let out = build_vault(&dir);
    assert_eq!(out.report.recipes_imported, 0);
    assert_eq!(out.report.warnings.len(), 1, "{:?}", out.report.warnings);
    assert!(
        out.report.warnings[0].contains("only.md")
            && out.report.warnings[0].contains("group description"),
        "{}",
        out.report.warnings[0]
    );
}

#[test]
fn a_file_that_fails_validation_is_skipped_and_its_siblings_load() {
    let dir = carrying(
        &[
            ("bad.md", "---\ndescription: no name at all\n---\nbody"),
            ("one.md", SKILL),
        ],
        &[
            (
                "bad.md",
                "---\nrecipe: vault\nname: nope\n---\n\nno fence here",
            ),
            ("one.md", RECIPE),
        ],
    );
    let out = build_vault(&dir);
    assert_eq!(out.report.skills_imported, 1, "the good one still loaded");
    assert_eq!(out.report.recipes_imported, 1);
    assert_eq!(out.report.warnings.len(), 2, "{:?}", out.report.warnings);
    assert!(
        out.report.warnings[0].contains(".kglite/skills/bad.md")
            && out.report.warnings[0].contains("name"),
        "{}",
        out.report.warnings[0]
    );
    assert!(
        out.report.warnings[1].contains(".kglite/recipes/bad.md")
            && out.report.warnings[1].contains("cypher"),
        "{}",
        out.report.warnings[1]
    );
    assert!(
        out.report.errors.is_empty(),
        "a skipped file is not an error"
    );
}

#[test]
fn a_recipe_file_needs_exactly_one_cypher_fence() {
    let two = RECIPE.replace(
        "```cypher\nMATCH (n) WHERE n.title = $title RETURN n.concept_id AS id\n```",
        "```cypher\nMATCH (n) RETURN n\n```\n\n```cypher\nMATCH (m) RETURN m\n```",
    );
    let err = crate::graph::recipes::parse_markdown(&two).unwrap_err();
    assert!(err.to_string().contains("found 2"), "{err}");
    // A non-cypher fence is not the statement.
    let other = RECIPE.replace("```cypher", "```python");
    let err = crate::graph::recipes::parse_markdown(&other).unwrap_err();
    assert!(err.to_string().contains("found none"), "{err}");
}

#[test]
fn a_recipe_file_may_omit_the_group_description_and_the_parameters() {
    let first = RECIPE.replace("recipe_description: Reading the vault.\n", "");
    let second = "---\nrecipe: vault\nname: everything\ndescription: Every note.\n\
                  recipe_description: Reading the vault.\n---\n\n\
                  ```cypher\nMATCH (n) RETURN n.concept_id AS id\n```\n";
    // Sorted order puts `a.md` first, so the group description is inherited
    // from a sibling in the same import.
    let dir = carrying(&[], &[("a.md", second), ("b.md", &first)]);
    let out = build_vault(&dir);
    assert_eq!(out.report.recipes_imported, 2, "{:?}", out.report.warnings);
    let inherited = crate::graph::recipes::get(&out.graph, "vault", "by_title").unwrap();
    assert_eq!(inherited.recipe_description, "Reading the vault.");
    let bare = crate::graph::recipes::get(&out.graph, "vault", "everything").unwrap();
    assert_eq!(
        bare.parameters["additionalProperties"], false,
        "the closed, parameter-free schema"
    );
}

#[test]
fn carried_directories_are_only_read_under_the_vault_dialect() {
    let dir = carrying(&[("one.md", SKILL)], &[("one.md", RECIPE)]);
    let out = build_as(&dir, Dialect::Loose).unwrap();
    assert_eq!(out.report.skills_imported, 0);
    assert_eq!(out.report.recipes_imported, 0);
    assert!(!out.graph.has_node_type("KgliteSkill"));
}

#[test]
fn a_rendered_recipe_parses_back_to_the_same_record() {
    // The writer the exporter uses (VAULT.md §10, §8) and the reader P6 added
    // are one dialect, so a vault written from a graph re-imports unchanged.
    let original = crate::graph::recipes::parse_markdown(RECIPE).unwrap();
    let rendered = crate::graph::recipes::render_markdown(&original);
    let back = crate::graph::recipes::parse_markdown(&rendered).unwrap();
    assert_eq!(back, original);
}

/// `structure:` is a profile override like `hubs:` — it has to reach the
/// *parse*, because the derivation runs there, over the tree the link pass
/// already read (VAULT.md §7.1).
#[test]
fn structure_reaches_the_notes_through_the_config() {
    let dir = vault_with(
        Some("kglite_vault: 1\nstructure:\n  sections: {}\n"),
        &[("note.md", "# One\n\ntext\n")],
    );
    let out = build_vault(&dir);
    assert!(
        labels(&out).contains(&("Section".to_string(), "note#One".to_string())),
        "{:?}",
        labels(&out)
    );
}

/// A `structure:` block a build cannot read fails it, exactly as any other
/// schema failure does — the file is what a rebuild re-applies, so a vault
/// whose rules stopped parsing would quietly lose every derived node.
#[test]
fn a_structure_block_this_build_cannot_read_fails_the_build() {
    let dir = vault_with(
        Some("kglite_vault: 1\nstructure:\n  paragraphs: {}\n"),
        &[("note.md", "# One\n")],
    );
    let message = match build_as(&dir, Dialect::Obsidian) {
        Err(message) => message,
        Ok(_) => panic!("the build fails on a rule it cannot read"),
    };
    assert!(
        message.contains("unknown key `structure.paragraphs`"),
        "{message}"
    );
    // …and `okf.validate` reports the same failure as the §9 error.
    let report =
        crate::okf::validate(dir.path(), &BuildOptions::for_dialect(Dialect::Obsidian)).unwrap();
    assert_eq!(report.errors.len(), 1);
}

/// The one compatibility promise: a vault that declares nothing derives
/// nothing, so every graph built before this feature is the graph it was.
#[test]
fn a_vault_without_the_block_derives_nothing() {
    let dir = vault_with(
        Some("kglite_vault: 1\n"),
        &[("note.md", "# One\n\ntext\n\n## Two\n")],
    );
    let out = build_vault(&dir);
    assert_eq!(
        out.report.nodes_by_label.keys().collect::<Vec<_>>(),
        vec!["Note"],
        "one note, and nothing else"
    );
}

/// `edge_defaults:` is a top-level key like `structure:` — declared in the
/// file, applied where the rows are emitted (VAULT.md §7.2).
#[test]
fn edge_defaults_reach_every_edge_of_their_type() {
    let dir = vault_with(
        Some("kglite_vault: 1\nedge_defaults:\n  LINKS_TO: {derivation: prose_reference}\n"),
        &[("a.md", "See [[b]].\n"), ("b.md", "# B\n")],
    );
    let out = build_vault(&dir);
    let props: Vec<(String, String)> = crate::okf::build::tests_support::edges_of(&out.graph)
        .into_iter()
        .filter(|(_, conn, _, _)| conn == "LINKS_TO")
        .flat_map(|(_, _, _, props)| props)
        .collect();
    assert!(
        props.contains(&("derivation".to_string(), "prose_reference".to_string())),
        "{props:?}"
    );
}

#[test]
fn an_edge_default_that_is_not_a_scalar_fails_the_build() {
    let dir = vault_with(
        Some("kglite_vault: 1\nedge_defaults:\n  LINKS_TO: {sources: [a, b]}\n"),
        &[("a.md", "# A\n")],
    );
    let message = match build_as(&dir, Dialect::Obsidian) {
        Err(message) => message,
        Ok(_) => panic!("a constant is a scalar"),
    };
    assert!(
        message.contains("`edge_defaults.LINKS_TO.sources` must be a scalar"),
        "{message}"
    );
}

/// `export:` (VAULT.md §7.3). Read by the exporter, not by the build — but
/// validated *here*, because an unknown key inside it is a compatibility
/// boundary exactly as `structure:`'s is, and a declaration nobody reads is
/// exactly what a typo produces.
#[test]
fn an_export_block_is_read_and_its_unknown_sub_key_refused() {
    let config =
        parse("kglite_vault: 1\nexport:\n  edge_tables:\n    WORKED_ON_BY: Worked on by\n")
            .expect("a declared edge table parses");
    assert_eq!(
        config.export_edge_tables,
        BTreeMap::from([("WORKED_ON_BY".to_string(), "Worked on by".to_string())])
    );

    let message = parse("kglite_vault: 1\nexport:\n  edge_table: {A: B}\n")
        .expect_err("a key this build does not read is refused by name");
    assert!(
        message.contains("unknown key `export.edge_table`"),
        "{message}"
    );
}

/// The two rules a declared entry has to satisfy: the key spells an edge type
/// and the value names a heading. Neither is checkable later — a lowercased
/// type would simply match no edge, and an empty heading would write the table
/// under `## `.
#[test]
fn a_declared_edge_table_names_an_edge_type_and_a_heading() {
    let lowercase = parse("kglite_vault: 1\nexport:\n  edge_tables:\n    worked_on_by: Who\n")
        .expect_err("an edge type is UPPER_SNAKE");
    assert!(
        lowercase.contains("`export.edge_tables.worked_on_by` is not an edge type"),
        "{lowercase}"
    );
    let blank = parse("kglite_vault: 1\nexport:\n  edge_tables:\n    WORKED_ON_BY: '  '\n")
        .expect_err("a heading is not blank");
    assert!(
        blank.contains("`export.edge_tables.WORKED_ON_BY` names no heading"),
        "{blank}"
    );
}

// ---------------------------------------------------------------------------
// `tag_labels:` (VAULT.md §5.5, §7)
// ---------------------------------------------------------------------------

/// The pattern is the whole schema: `<prefix>/*` and nothing else. A rule
/// spelled any other way would match no tag and say nothing, which is the
/// reassuring-direction failure a config error exists to prevent.
#[test]
fn tag_labels_accepts_a_prefix_rule_and_refuses_every_other_pattern() {
    let config =
        parse("kglite_vault: 1\ntag_labels:\n  \"intent/*\": {label: Intent, edge: HAS_INTENT}\n")
            .expect("the documented shape parses");
    assert_eq!(
        config.tag_labels.get("intent/*"),
        Some(&crate::okf::model::TagLabelSpec {
            prefix: "intent/".to_string(),
            label: "Intent".to_string(),
            edge: "HAS_INTENT".to_string(),
        })
    );
    for pattern in ["intent", "intent/", "*", "*/intent", "intent/*/*", "in*t/*"] {
        let message = parse(&format!(
            "kglite_vault: 1\ntag_labels:\n  \"{pattern}\": {{label: Intent, edge: HAS_INTENT}}\n"
        ))
        .expect_err("only `<prefix>/*` is a tag pattern");
        assert!(
            message.contains(&format!("`tag_labels.{pattern}`")),
            "the message names the key that is wrong: {message}"
        );
    }
}

/// Both fields are required, and each is spelled the way its kind is spelled
/// everywhere else in this file.
#[test]
fn a_tag_label_rule_names_a_label_and_an_edge_type() {
    let no_edge = parse("kglite_vault: 1\ntag_labels:\n  \"intent/*\": {label: Intent}\n")
        .expect_err("a rule with no edge states no relationship");
    assert!(no_edge.contains("`tag_labels.intent/*.edge`"), "{no_edge}");

    let no_label = parse("kglite_vault: 1\ntag_labels:\n  \"intent/*\": {edge: HAS_INTENT}\n")
        .expect_err("a rule with no label names no node");
    assert!(
        no_label.contains("`tag_labels.intent/*.label`"),
        "{no_label}"
    );

    let lowercase =
        parse("kglite_vault: 1\ntag_labels:\n  \"intent/*\": {label: Intent, edge: has_intent}\n")
            .expect_err("an edge type is UPPER_SNAKE");
    assert!(
        lowercase.contains("`tag_labels.intent/*.edge` is not an edge type"),
        "{lowercase}"
    );

    let unknown = parse(
        "kglite_vault: 1\ntag_labels:\n  \
         \"intent/*\": {label: Intent, edge: HAS_INTENT, case_insensitive: false}\n",
    )
    .expect_err("a field this build does not read is refused by name");
    assert!(
        unknown.contains("unknown key `tag_labels.intent/*.case_insensitive`"),
        "{unknown}"
    );
}
