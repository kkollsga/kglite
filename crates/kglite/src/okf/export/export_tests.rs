//! Export tests (VAULT.md §10).

use super::*;
use crate::okf::model::{BuildOptions, Dialect};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

fn golden_vault() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/okf/golden/vault")
}

fn build_vault(dir: &Path) -> Arc<DirGraph> {
    crate::okf::build(dir, &BuildOptions::for_dialect(Dialect::Obsidian))
        .unwrap()
        .graph
}

/// Build a throwaway vault from `(relative path, contents)` pairs.
fn write_vault(root: &Path, files: &[(&str, &str)]) {
    for (rel, text) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

/// Every file under `dir`, vault-relative, sorted — directories are not listed.
fn tree(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_tree(dir, dir, &mut out);
    out.sort();
    out
}

fn collect_tree(root: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_tree(root, &path, out);
        } else {
            out.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

/// Every file's bytes, vault-relative — the whole tree as one comparable value.
fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    tree(dir)
        .into_iter()
        .map(|rel| {
            let bytes = std::fs::read(dir.join(&rel)).unwrap();
            (rel, bytes)
        })
        .collect()
}

fn read(dir: &Path, rel: &str) -> String {
    std::fs::read_to_string(dir.join(rel)).unwrap_or_else(|e| panic!("reading {rel}: {e}"))
}

fn export_to(graph: &DirGraph, dir: &Path) -> ExportReport {
    export(graph, dir, &ExportOptions::default()).unwrap()
}

/// A one-note vault with the frontmatter a test wants, exported.
fn export_one(front: &str, body: &str) -> (tempfile::TempDir, String) {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[("notes/subject.md", &format!("{front}{body}"))],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let files = tree(out.path());
    let note = files
        .iter()
        .find(|f| f.ends_with("subject.md"))
        .unwrap_or_else(|| panic!("no subject.md in {files:?}"))
        .clone();
    let text = read(out.path(), &note);
    (out, text)
}

// ── §10.1 which nodes become files ─────────────────────────────────────────

#[test]
fn synthesized_hub_folder_and_stub_nodes_are_not_files() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[(
            "one.md",
            "---\ntags:\n  - alpha\n---\nLinks to [[Nowhere]] and https://example.com/x.\n",
        )],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(
        tree(out.path()),
        vec![".kglite/export-manifest.json", "Note/one.md"],
        "the Tag hub, the Source URL, the Folder and the dangling stub all regenerate"
    );
}

#[test]
fn the_carried_skill_and_recipe_land_under_kglite() {
    let graph = build_vault(&golden_vault());
    let out = tempfile::tempdir().unwrap();
    let report = export_to(&graph, out.path());
    assert_eq!((report.skills_written, report.recipes_written), (1, 1));
    let skill = read(out.path(), ".kglite/skills/vault_overview.md");
    assert!(
        skill.starts_with("---\nname: \"vault_overview\""),
        "{skill}"
    );
    let recipe = read(out.path(), ".kglite/recipes/vault.by_keyword.md");
    assert!(
        recipe.contains("```cypher\nMATCH (n)-[:HAS_KEYWORD]"),
        "{recipe}"
    );
    assert!(
        recipe.contains("\"additionalProperties\":false"),
        "{recipe}"
    );
}

#[test]
fn a_carried_skill_and_recipe_reimport_from_the_export() {
    let graph = build_vault(&golden_vault());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let reimported = build_vault(out.path());
    let skills = crate::graph::skills::list(&reimported);
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "vault_overview");
    let recipes = crate::graph::recipes::list(&reimported);
    assert_eq!(recipes.len(), 1);
    assert_eq!(
        (recipes[0].recipe.as_str(), recipes[0].name.as_str()),
        ("vault", "by_keyword")
    );
    assert_eq!(
        recipes[0].recipe_description,
        "Navigating the golden vault."
    );
}

/// Add `ids` as nodes of `label`, with `title` = the id.
fn add_nodes(graph: &mut DirGraph, label: &str, ids: &[&str]) {
    let frame = crate::datatypes::values::DataFrame::from_cypher_rows(
        vec!["concept_id".to_string(), "title".to_string()],
        ids.iter()
            .map(|id| {
                vec![
                    Value::String((*id).to_string()),
                    Value::String((*id).to_string()),
                ]
            })
            .collect(),
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        graph,
        frame,
        label.to_string(),
        "concept_id".to_string(),
        Some("title".to_string()),
        Some("update".to_string()),
    )
    .unwrap();
}

#[test]
fn a_graph_that_never_had_files_exports_every_node_but_the_system_labels() {
    // No `file_path` anywhere, so "no provenance" cannot mean "synthesized" —
    // the label list is the only thing keeping the system labels out.
    let mut graph = DirGraph::new();
    add_nodes(&mut graph, "Person", &["a", "b"]);
    add_nodes(&mut graph, TAG_LABEL, &["seismic"]);
    add_nodes(&mut graph, SOURCE_LABEL, &["https://example.com/x"]);
    add_nodes(&mut graph, FOLDER_LABEL, &["notes"]);
    add_nodes(&mut graph, IMAGE_LABEL, &["img/x.png"]);
    add_nodes(&mut graph, ATTACHMENT_LABEL, &["img/x.pdf"]);
    crate::graph::skills::set(
        &mut graph,
        &crate::graph::skills::SkillRecord {
            name: "how_to".to_string(),
            description: "What this graph holds.".to_string(),
            body: "Query it with Cypher.".to_string(),
            references_tools: Vec::new(),
            delivery: crate::graph::skills::Delivery::default(),
        },
    )
    .unwrap();

    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(
        tree(out.path()),
        vec![
            ".kglite/export-manifest.json",
            ".kglite/skills/how_to.md",
            "Person/a.md",
            "Person/b.md"
        ],
        "the system labels are never note files, whatever the graph's provenance"
    );
}

// ── §10.2 file paths ───────────────────────────────────────────────────────

#[test]
fn a_path_whose_top_folder_is_the_label_is_preserved() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("Note/kept.md", "Stays where it is.\n"),
            ("Note/deep/also.md", "Top folder still matches.\n"),
            ("elsewhere/moved.md", "---\ntype: Note\n---\nRe-filed.\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let files = tree(out.path());
    assert!(files.contains(&"Note/kept.md".to_string()), "{files:?}");
    assert!(
        files.contains(&"Note/deep/also.md".to_string()),
        "{files:?}"
    );
    assert!(files.contains(&"Note/moved.md".to_string()), "{files:?}");
    assert!(
        !files.contains(&"elsewhere/moved.md".to_string()),
        "{files:?}"
    );
}

#[test]
fn a_folder_note_keeps_its_spelling_beside_its_folder() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("Note/team.md", "The folder note for `Note/team/`.\n"),
            ("Note/team/member.md", "Inside it.\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let files = tree(out.path());
    assert!(
        files.contains(&"Note/team.md".to_string())
            && files.contains(&"Note/team/member.md".to_string()),
        "the folder note is not moved into its own folder: {files:?}"
    );
}

/// A note whose folder cannot be preserved still keeps its filename: the stem
/// is the link namespace (§3, §5.2), so renaming it after the title dangles
/// every body wikilink that spelled the old one.
#[test]
fn a_relocated_note_keeps_the_stem_its_links_name() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("titled.md", "---\ntitle: A Nice Title\n---\nprose\n"),
            ("plain.md", "prose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(
        tree(out.path()),
        vec![
            ".kglite/export-manifest.json",
            "Note/plain.md",
            "Note/titled.md"
        ],
        "only the folder moves"
    );
    assert!(
        read(out.path(), "Note/titled.md").contains("title: A Nice Title\n"),
        "and the title the stem does not spell is written out"
    );
}

/// The title-then-id rung, which only a node with no file behind it reaches.
#[test]
fn a_note_no_file_ever_backed_is_named_by_its_title_then_its_id() {
    let graph = titled_notes(&[("titled", "A Nice Title"), ("plain", "")]);
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(
        tree(out.path()),
        vec![
            ".kglite/export-manifest.json",
            "Note/A Nice Title.md",
            "Note/plain.md"
        ]
    );
}

/// Sanitisation bites on a *generated* name only: a preserved stem came from a
/// real file, where the filesystem already refused these characters.
#[test]
fn forbidden_characters_in_a_generated_name_are_replaced() {
    let graph = titled_notes(&[("n", "a/b:c*d?e\"f<g>h|i")]);
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(
        tree(out.path()),
        vec![".kglite/export-manifest.json", "Note/a-b-c-d-e-f-g-h-i.md"]
    );
}

/// A graph no file ever backed: nothing here carries `file_path`, so every node
/// is a note (§10.1) reaching §10.2's title-then-id rung. An empty title is how
/// a test asks for the id rung.
fn titled_notes(pairs: &[(&str, &str)]) -> DirGraph {
    let mut graph = DirGraph::new();
    let frame = crate::datatypes::values::DataFrame::from_cypher_rows(
        vec!["concept_id".to_string(), "title".to_string()],
        pairs
            .iter()
            .map(|(id, title)| {
                vec![
                    Value::String((*id).to_string()),
                    Value::String((*title).to_string()),
                ]
            })
            .collect(),
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        frame,
        "Note".to_string(),
        "concept_id".to_string(),
        Some("title".to_string()),
        Some("update".to_string()),
    )
    .unwrap();
    graph
}

#[test]
fn a_case_only_path_collision_appends_the_id() {
    // `Roadmap.md` and `roadmap.md` are one file on macOS and Windows — and
    // both stems are preserved from their source files, so the suffix rule is
    // what keeps the two apart.
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("a/Roadmap.md", "---\ntype: Doc\n---\nUpper.\n"),
            ("b/roadmap.md", "---\ntype: Doc\n---\nLower.\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(
        tree(out.path()),
        vec![
            ".kglite/export-manifest.json",
            "Doc/Roadmap.md",
            "Doc/roadmap-roadmap.md"
        ],
        "the second note in (label, id) order takes the suffix"
    );
}

// ── §10.3 / §10.4 frontmatter ──────────────────────────────────────────────

#[test]
fn type_is_never_emitted() {
    let (_dir, text) = export_one("---\ntype: Report\n---\n", "prose\n");
    assert!(!text.contains("type:"), "{text}");
}

#[test]
fn id_and_title_are_emitted_only_when_the_stem_does_not_say_them() {
    let (_dir, silent) = export_one("---\ntitle: subject\n---\n", "prose\n");
    assert_eq!(silent, "prose\n", "stem == id == title, so nothing to say");

    let (_dir, spoken) = export_one("---\nid: other-id\ntitle: Other Title\n---\n", "prose\n");
    assert!(spoken.contains("id: other-id\n"), "{spoken}");
    assert!(spoken.contains("title: Other Title\n"), "{spoken}");
}

#[test]
fn dotted_keys_become_nested_maps_again() {
    let (_dir, text) = export_one(
        "---\nmetadata:\n  source: vendor\n  depth: 2\n---\n",
        "prose\n",
    );
    assert!(
        text.contains("metadata:\n  depth: 2\n  source: vendor\n"),
        "{text}"
    );
}

#[test]
fn lists_become_yaml_sequences() {
    let (_dir, text) = export_one("---\ntags:\n  - one\n  - two\n---\n", "prose\n");
    assert!(text.contains("tags:\n  - one\n  - two\n"), "{text}");
}

#[test]
fn keys_are_sorted() {
    let (_dir, text) = export_one("---\nzebra: 1\nalpha: 2\nmid: 3\n---\n", "prose\n");
    let keys: Vec<&str> = text
        .lines()
        .filter(|l| l.contains(": "))
        .map(|l| l.split(':').next().unwrap())
        .collect();
    assert_eq!(keys, vec!["alpha", "mid", "zebra"], "{text}");
}

#[test]
fn temporal_values_become_iso_strings() {
    let (_dir, text) = export_one(
        "---\nupdated: 2026-01-15\nreviewed: '2026-01-15T09:30:00Z'\n---\n",
        "prose\n",
    );
    assert!(text.contains("updated: 2026-01-15\n"), "{text}");
    assert!(text.contains("reviewed: 2026-01-15T09:30:00Z\n"), "{text}");
}

#[test]
fn a_point_becomes_wkt() {
    let mut graph = DirGraph::new();
    let frame = crate::datatypes::values::DataFrame::from_cypher_rows(
        vec![
            "concept_id".to_string(),
            "title".to_string(),
            "where".to_string(),
        ],
        vec![vec![
            Value::String("p".to_string()),
            Value::String("p".to_string()),
            Value::Point {
                lat: 60.5,
                lon: 5.25,
            },
        ]],
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        frame,
        "Site".to_string(),
        "concept_id".to_string(),
        Some("title".to_string()),
        Some("update".to_string()),
    )
    .unwrap();
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert!(
        read(out.path(), "Site/p.md").contains("where: POINT(5.25 60.5)"),
        "{}",
        read(out.path(), "Site/p.md")
    );
}

#[test]
fn the_body_property_is_never_frontmatter() {
    let (_dir, text) = export_one("---\ntitle: subject\n---\n", "the prose\n");
    assert!(!text.contains("body:"), "{text}");
    assert!(text.ends_with("the prose\n"), "{text}");
}

// ── §10.4 quoting ──────────────────────────────────────────────────────────

#[test]
fn a_string_that_would_reparse_as_something_else_is_quoted() {
    // Each of these is a *string* in the graph — a scalar because `types:`
    // declared it so, a list element because §4.2's inference never reaches
    // inside a sequence. Unquoted, the next import hands back another type.
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            (
                ".kglite/vault.yaml",
                "kglite_vault: 1\ntypes:\n  Note:\n    numeric: string\n    floaty: string\n    \
                 flag: string\n    empty: string\n    dated: string\n",
            ),
            (
                "n.md",
                "---\nnumeric: \"12\"\nfloaty: \"1.5\"\nflag: \"true\"\nempty: \"\"\n\
                 dated: \"2026-01-15\"\ncolon: \"a: b\"\nhashed: \"a #b\"\n\
                 leading: \"- dash\"\n\
                 mixed: [\"[[A]]\", \"plain\", 2026-01-15, \"12\", \"true\"]\n---\nprose\n",
            ),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let text = read(out.path(), "Note/n.md");
    for expected in [
        "numeric: \"12\"",
        "floaty: \"1.5\"",
        "flag: \"true\"",
        "empty: \"\"",
        "dated: \"2026-01-15\"",
        "colon: \"a: b\"",
        "hashed: \"a #b\"",
        "leading: \"- dash\"",
        "  - \"[[A]]\"",
        "  - plain",
        "  - \"2026-01-15\"",
    ] {
        assert!(text.contains(expected), "missing {expected} in\n{text}");
    }
    // And the round trip proves the rule rather than the spelling. `dated` is
    // not in this list on purpose: §4.2 says quoting alone does not stop
    // temporal inference, so a top-level date-like string needs the `types:`
    // declaration the export does not write — a documented loss (§10.9), not a
    // quoting failure. Inside a sequence inference never runs, so `mixed`
    // below keeps its date-like element a string.
    let back = build_vault(out.path());
    for (key, expected) in [
        ("numeric", "12"),
        ("floaty", "1.5"),
        ("flag", "true"),
        ("colon", "a: b"),
        ("hashed", "a #b"),
        ("leading", "- dash"),
    ] {
        assert_eq!(
            property(&back, "n", key),
            Some(Value::String(expected.to_string())),
            "{key} did not come back a string"
        );
    }
    assert_eq!(
        property(&back, "n", "mixed"),
        Some(Value::List(
            ["[[A]]", "plain", "2026-01-15", "12", "true"]
                .into_iter()
                .map(|s| Value::String(s.to_string()))
                .collect()
        )),
        "a mixed list keeps every element a string"
    );
}

/// One node's property, by id.
fn property(graph: &DirGraph, id: &str, key: &str) -> Option<Value> {
    graph.graph.node_indices().find_map(|idx| {
        let view = graph.node_view(idx)?;
        (view.id().as_ref() == &Value::String(id.to_string()))
            .then(|| view.get_property_value(key))
            .flatten()
    })
}

// ── §10.5 body ─────────────────────────────────────────────────────────────

#[test]
fn the_body_is_verbatim_and_a_bodyless_note_is_frontmatter_only() {
    let body = "# Heading\n\nSome *prose* with  odd   spacing.\n";
    let (_dir, text) = export_one("---\nkeep: 1\n---\n", body);
    assert!(text.ends_with(body), "{text}");

    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), &[("n.md", "---\nkeep: 1\n---\n")]);
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert_eq!(read(out.path(), "Note/n.md"), "---\nkeep: 1\n---\n");
}

// ── §10.6 edges ────────────────────────────────────────────────────────────

#[test]
fn typed_edges_become_sorted_wikilink_lists() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            (
                "a.md",
                "---\ndepends_on:\n  - \"[[c]]\"\n  - \"[[b]]\"\n---\nprose\n",
            ),
            ("b.md", "prose\n"),
            ("c.md", "prose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert!(
        read(out.path(), "Note/a.md").contains("depends_on:\n  - \"[[b]]\"\n  - \"[[c]]\"\n"),
        "{}",
        read(out.path(), "Note/a.md")
    );
}

#[test]
fn a_link_already_in_the_body_is_not_repeated_in_frontmatter() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("a.md", "Body links to [[b]] and embeds ![[c]].\n"),
            ("b.md", "prose\n"),
            ("c.md", "prose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let text = read(out.path(), "Note/a.md");
    assert!(!text.contains("links_to:"), "{text}");
    assert!(!text.contains("embeds:"), "{text}");
    assert_eq!(text, "Body links to [[b]] and embeds ![[c]].\n");
}

#[test]
fn a_typed_edge_is_emitted_even_when_the_body_links_the_same_target() {
    // `DEPENDS_ON` says something `[[b]]` does not; dropping it would retype
    // the edge to LINKS_TO on the next import.
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("a.md", "---\ndepends_on: \"[[b]]\"\n---\nAlso see [[b]].\n"),
            ("b.md", "prose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    assert!(
        read(out.path(), "Note/a.md").contains("depends_on:\n  - \"[[b]]\"\n"),
        "{}",
        read(out.path(), "Note/a.md")
    );
}

#[test]
fn structural_and_synthesized_edges_are_not_emitted() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[(
            "a.md",
            "---\ntags:\n  - alpha\n---\nSee https://example.com/x.\n",
        )],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let text = read(out.path(), "Note/a.md");
    assert!(!text.contains("tagged:"), "{text}");
    assert!(!text.contains("contains:"), "{text}");
    assert!(
        !text.contains("links_to:"),
        "the Source URL is not a file: {text}"
    );
    assert!(
        text.contains("tags:\n  - alpha\n"),
        "the property stays: {text}"
    );
}

#[test]
fn an_ambiguous_target_is_written_folder_qualified() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            (
                "start.md",
                "---\ntype: Start\nsee_also: \"[[one/dup]]\"\n---\nprose\n",
            ),
            ("one/dup.md", "---\ntype: One\n---\nprose\n"),
            ("two/dup.md", "---\ntype: Two\n---\nprose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let text = read(out.path(), "Start/start.md");
    assert!(text.contains("see_also:\n  - \"[[One/dup]]\"\n"), "{text}");
}

#[test]
fn a_parent_the_layout_no_longer_expresses_is_emitted_as_its_edge_key() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("hub.md", "---\ntitle: Hub\n---\nprose\n"),
            ("leaf.md", "---\nparent: \"[[hub]]\"\n---\nprose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let text = read(out.path(), "Note/leaf.md");
    // `[[hub]]`, not `[[Hub]]`: the re-filed note kept the stem the link names.
    assert!(text.contains("child_of:\n  - \"[[hub]]\"\n"), "{text}");
    // And it comes back as the same edge type, not as a `parent` property.
    let back = build_vault(out.path());
    assert!(
        edge_types(&back).contains(&"CHILD_OF".to_string()),
        "{:?}",
        edge_types(&back)
    );
}

fn edge_types(graph: &DirGraph) -> Vec<String> {
    let mut out: Vec<String> = graph
        .graph
        .edge_indices()
        .filter_map(|e| {
            graph
                .graph
                .edge_weight(e)
                .map(|d| d.connection_type_str(&graph.interner).to_string())
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

#[test]
fn edge_properties_are_dropped_and_counted() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("a.md", "---\ntitle: a\n---\n## Deep\n\n[[b#anchor]]\n"),
            ("b.md", "prose\n"),
        ],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    let report = export_to(&graph, out.path());
    assert_eq!(
        report.edge_properties_dropped, 2,
        "`section` and `anchor` on the one body link"
    );
    let text = read(out.path(), "Note/a.md");
    assert!(
        !text.contains("links_to"),
        "the link is already in the body, so no edge key repeats it: {text}"
    );
}

// ── §10.5 the frontmatter/body seam ────────────────────────────────────────

#[test]
fn no_separator_line_is_inserted_between_the_frontmatter_and_the_body() {
    // The reader keeps everything below the closing `---`, so a blank line
    // written here comes back as a leading newline on `body` and the next
    // export writes another one — one line of growth per round trip.
    let (_dir, tight) = export_one("---\nid: x\n---\n", "prose\n");
    assert!(tight.ends_with("---\nprose\n"), "{tight}");
    let (_dir, spaced) = export_one("---\nid: x\n---\n", "\nprose\n");
    assert!(
        spaced.ends_with("---\n\nprose\n"),
        "the author's own blank line is body, and survives: {spaced}"
    );
}

// ── §10.3 the title the next import recovers ───────────────────────────────

#[test]
fn a_title_a_heading_would_shadow_is_written_out() {
    // The §3 ladder reads the body's first heading *before* the filename stem,
    // so leaving `title:` out because the title equals the stem hands the next
    // import the heading instead.
    let (_dir, text) = export_one("---\ntitle: subject\n---\n", "## Deep dive\n\nprose\n");
    assert!(
        text.starts_with("---\ntitle: subject\n---\n"),
        "the heading would have taken the title: {text}"
    );
}

/// The §3 ladder reads `name:` before the heading and the stem, and `name:`
/// is an ordinary property the export writes out — so a note titled by one
/// needs no `title:` either.
#[test]
fn a_title_a_name_key_already_states_is_left_out() {
    let (_dir, text) = export_one("---\nname: Claude memory\n---\n", "## Deep dive\n\nprose\n");
    assert!(
        text.contains("name: Claude memory"),
        "`name:` is a property, not a reserved key: {text}"
    );
    assert!(
        !text.contains("title:"),
        "the reader reads `name:` before the heading: {text}"
    );
    let (dir, _) = export_one("---\nname: Claude memory\n---\n", "## Deep dive\n\nprose\n");
    let back = build_vault(dir.path());
    assert!(
        title_of(&back, "subject") == "Claude memory",
        "and recovers exactly that"
    );
}

/// A body link that resolves through the target's `aliases:` (§5.2 rung 3) is
/// the same statement as a plain one, so it must not be repeated as a
/// frontmatter key — the repeat becomes a second edge on the next import, one
/// carrying the body's `section` and one carrying nothing.
#[test]
fn a_body_link_resolved_through_an_alias_is_not_repeated_in_frontmatter() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("Note/a.md", "## Deep dive\n\nSee [[The Other One]].\n"),
            (
                "Note/b.md",
                "---\naliases:\n  - The Other One\n---\nprose\n",
            ),
        ],
    );
    let graph = build_vault(source.path());
    assert_eq!(links_to_count(&graph), 1);
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let text = read(out.path(), "Note/a.md");
    assert!(!text.contains("links_to"), "{text}");
    assert_eq!(
        links_to_count(&build_vault(out.path())),
        1,
        "and the re-import still has one edge, not the body's plus a copy"
    );
}

fn links_to_count(graph: &DirGraph) -> usize {
    graph
        .graph
        .edge_indices()
        .filter(|e| {
            graph
                .graph
                .edge_weight(*e)
                .is_some_and(|d| d.connection_type_str(&graph.interner) == "LINKS_TO")
        })
        .count()
}

fn title_of(graph: &DirGraph, id: &str) -> String {
    for idx in graph.graph.node_indices() {
        if let Some(view) = graph.node_view(idx) {
            if matches!(view.id().as_ref(), Value::String(s) if s == id) {
                return crate::datatypes::values::raw_string(&view.title());
            }
        }
    }
    panic!("no node with id `{id}`");
}

#[test]
fn a_title_the_reader_recovers_on_its_own_is_left_out() {
    let (_dir, heading) = export_one("---\ntitle: Deep dive\n---\n", "## Deep dive\n\nprose\n");
    assert!(
        !heading.contains("title:"),
        "the body's own heading already says it: {heading}"
    );
    let (_dir, stem) = export_one("---\ntitle: subject\n---\n", "prose with no heading\n");
    assert!(
        !stem.contains("title:"),
        "the filename stem already says it: {stem}"
    );
}

// ── §10.7 manifest safety ──────────────────────────────────────────────────

#[test]
fn the_manifest_names_every_file_the_export_wrote() {
    let graph = build_vault(&golden_vault());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let manifest: serde_json::Value =
        serde_json::from_str(&read(out.path(), ".kglite/export-manifest.json")).unwrap();
    assert_eq!(manifest["kglite_vault"], 1);
    let files = manifest["files"].as_object().unwrap();
    let mut listed: Vec<&String> = files.keys().collect();
    listed.sort();
    let mut written = tree(out.path());
    written.retain(|f| f != ".kglite/export-manifest.json");
    assert_eq!(
        listed.into_iter().cloned().collect::<Vec<String>>(),
        written,
        "the manifest is not allowed to omit a file it wrote"
    );
}

#[test]
fn a_foreign_file_is_refused_and_force_replaces_it() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), &[("n.md", "prose\n")]);
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(out.path().join("Note")).unwrap();
    std::fs::write(out.path().join("Note/n.md"), "somebody else's file\n").unwrap();

    let report = export_to(&graph, out.path());
    assert_eq!((report.files_written, report.files_refused), (0, 1));
    assert_eq!(
        report.refusals,
        vec!["Note/n.md: not written by an export (use force to replace)"]
    );
    assert_eq!(read(out.path(), "Note/n.md"), "somebody else's file\n");

    let forced = export(
        &graph,
        out.path(),
        &ExportOptions {
            force: true,
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!((forced.files_written, forced.files_refused), (1, 0));
    assert_eq!(read(out.path(), "Note/n.md"), "prose\n");
}

#[test]
fn an_edited_owned_file_is_refused_and_stays_owned() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), &[("n.md", "prose\n")]);
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    std::fs::write(out.path().join("Note/n.md"), "a human wrote this\n").unwrap();

    let report = export_to(&graph, out.path());
    assert_eq!((report.files_written, report.files_refused), (0, 1));
    assert_eq!(
        report.refusals,
        vec!["Note/n.md: edited since the last export (use force to replace)"]
    );
    assert_eq!(read(out.path(), "Note/n.md"), "a human wrote this\n");
    // Still owned: an export of an empty graph must not now read it as a file
    // whose node disappeared and delete the edit it just protected.
    let empty = DirGraph::new();
    let after = export_to(&empty, out.path());
    assert_eq!((after.files_deleted, after.files_refused), (0, 1));
    assert_eq!(read(out.path(), "Note/n.md"), "a human wrote this\n");
}

#[test]
fn an_unchanged_file_is_not_rewritten() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), &[("n.md", "prose\n")]);
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    let before = std::fs::metadata(out.path().join("Note/n.md"))
        .unwrap()
        .modified()
        .unwrap();

    let again = export_to(&graph, out.path());
    assert_eq!((again.files_written, again.files_unchanged), (0, 1));
    let after = std::fs::metadata(out.path().join("Note/n.md"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(
        before, after,
        "an unchanged file keeps its modification time"
    );
}

#[test]
fn only_manifest_owned_files_are_deleted() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[("gone.md", "prose\n"), ("stays.md", "prose\n")],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    export_to(&graph, out.path());
    std::fs::write(out.path().join("Note/theirs.md"), "not ours\n").unwrap();

    let smaller = tempfile::tempdir().unwrap();
    write_vault(smaller.path(), &[("stays.md", "prose\n")]);
    let report = export_to(&build_vault(smaller.path()), out.path());
    assert_eq!(report.files_deleted, 1);
    let files = tree(out.path());
    assert!(!files.contains(&"Note/gone.md".to_string()), "{files:?}");
    assert!(files.contains(&"Note/stays.md".to_string()), "{files:?}");
    assert!(
        files.contains(&"Note/theirs.md".to_string()),
        "a file the manifest never named is never deleted: {files:?}"
    );
}

#[test]
fn a_manifest_of_an_unknown_version_is_an_error() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), &[("n.md", "prose\n")]);
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(out.path().join(".kglite")).unwrap();
    std::fs::write(
        out.path().join(".kglite/export-manifest.json"),
        "{\"kglite_vault\": 99, \"files\": {}}",
    )
    .unwrap();
    let error = export(&graph, out.path(), &ExportOptions::default()).unwrap_err();
    assert!(error.contains("kglite_vault: 99"), "{error}");
}

// ── §10.9 attachments ──────────────────────────────────────────────────────

#[test]
fn attachments_are_copied_when_the_source_root_is_known() {
    let graph = build_vault(&golden_vault());
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(golden_vault()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        (report.attachments_copied, report.attachments_unresolved),
        (3, 0)
    );
    for rel in ["img/diagram.png", "img/faults.png", "img/handbook.pdf"] {
        assert!(out.path().join(rel).is_file(), "{rel} was not copied");
    }
}

#[test]
fn attachments_are_reported_unresolved_without_a_source_root() {
    // A graph with no provenance to fall back on: nothing says where the
    // pictures are, so the references are counted and no bytes travel.
    let mut graph = (*build_vault(&golden_vault())).clone();
    graph.source_root = None;
    let out = tempfile::tempdir().unwrap();
    let report = export_to(&graph, out.path());
    assert_eq!(
        (report.attachments_copied, report.attachments_unresolved),
        (0, 3)
    );
    assert!(!out.path().join("img").exists());
}

/// A vault-built graph knows where its own pictures are (VAULT.md §12), so an
/// export that is told nothing still copies them. Repeating the path at the
/// call site is how a caller and the graph come to disagree about it.
#[test]
fn a_vault_built_graph_exports_its_attachments_without_being_told_the_root() {
    let graph = build_vault(&golden_vault());
    assert!(graph.source_root.is_some(), "the build stamped a root");
    let out = tempfile::tempdir().unwrap();
    let report = export_to(&graph, out.path());
    assert_eq!(
        (report.attachments_copied, report.attachments_unresolved),
        (3, 0)
    );
    assert!(out.path().join("img/faults.png").is_file());
}

/// A vault whose one note reaches one image note-relative, so an export told
/// the source root has exactly one attachment to copy.
const MEDIA: &[(&str, &str)] = &[
    (
        "Media/figures.md",
        "---\ntitle: Figures\n---\n![Fault map](../img/faults.png) is note-relative.\n",
    ),
    ("img/faults.png", "stand-in bytes\n"),
];

/// 2020-01-01T00:00:00Z. The source file is backdated to it so that "the copy
/// kept the source's modification time" is a claim the clock cannot satisfy by
/// accident: an export that stamped the destination with the time of the copy
/// passes a same-second comparison and fails this one.
fn fixed_mtime() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_577_836_800)
}

fn set_mtime(path: &Path, when: SystemTime) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(when)
        .unwrap();
}

fn modified(path: &Path) -> SystemTime {
    std::fs::metadata(path).unwrap().modified().unwrap()
}

/// A source vault with one backdated image, and the graph built from it.
fn backdated_media() -> (tempfile::TempDir, Arc<DirGraph>) {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), MEDIA);
    set_mtime(&source.path().join("img/faults.png"), fixed_mtime());
    let graph = build_vault(source.path());
    (source, graph)
}

/// `mtime` is a node property the loader reads back from `stat` (VAULT.md
/// §6.3), and §10.9 does not list it among the round trip's losses — so the
/// copy has to arrive carrying the source file's, not the time it was written.
#[test]
fn a_copied_attachment_keeps_the_source_files_modification_time() {
    let (source, graph) = backdated_media();
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        (report.attachments_copied, report.attachments_unresolved),
        (1, 0)
    );
    assert_eq!(
        modified(&out.path().join("img/faults.png")),
        fixed_mtime(),
        "the copy was stamped with the time of the copy, not the source's"
    );
}

/// The manifest's unchanged branch (§10.7) covers attachment bytes as well as
/// note bytes: a re-export of a vault nothing changed writes no file at all,
/// so ten thousand pictures are not rewritten to say the same thing.
#[test]
fn a_second_export_leaves_an_unchanged_attachment_alone() {
    let (source, graph) = backdated_media();
    let out = tempfile::tempdir().unwrap();
    let opts = ExportOptions {
        source_root: Some(source.path().to_path_buf()),
        ..ExportOptions::default()
    };
    let first = export(&graph, out.path(), &opts).unwrap();
    assert!(first.files_written > 0);
    let second = export(&graph, out.path(), &opts).unwrap();
    assert_eq!(
        (second.files_written, second.files_unchanged),
        (0, first.files_written),
        "the second export rewrote a file whose bytes had not moved"
    );
    assert_eq!(
        modified(&out.path().join("img/faults.png")),
        fixed_mtime(),
        "and left the modification time where the first export put it"
    );
}

// ── §10.8 determinism ──────────────────────────────────────────────────────

#[test]
fn exporting_twice_produces_identical_bytes() {
    let graph = build_vault(&golden_vault());
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let opts = ExportOptions {
        source_root: Some(golden_vault()),
        ..ExportOptions::default()
    };
    export(&graph, first.path(), &opts).unwrap();
    export(&build_vault(&golden_vault()), second.path(), &opts).unwrap();
    assert_eq!(
        snapshot(first.path()),
        snapshot(second.path()),
        "the same graph must produce the same tree, byte for byte"
    );
}

// ── the round trip (the first assertion; P10 makes it a corpus) ────────────

#[test]
fn the_golden_vault_round_trips_apart_from_its_documented_losses() {
    let source = golden_vault();
    let original = build_vault(&source);
    let out = tempfile::tempdir().unwrap();
    export(
        &original,
        out.path(),
        &ExportOptions {
            source_root: Some(source.clone()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let back = build_vault(out.path());

    // Notes keep their labels because the export files each one under its
    // label and §10.3 never writes `type:`; the re-import's folder rung
    // answers with the same word.
    assert_eq!(
        label_counts(&back).get("Article"),
        label_counts(&original).get("Article")
    );
    assert_eq!(
        label_counts(&back).get("Initiative"),
        label_counts(&original).get("Initiative")
    );
    // Documented losses, each with its reason:
    //  - `Keyword` needs the `hubs:` declaration, which lives in
    //    `.kglite/vault.yaml` and is not part of the graph;
    //  - `Folder` counts follow the new layout, which is one folder per label;
    //  - `Concept` is the dangling stub, which is re-made from the body.
    assert_eq!(label_counts(&back).get("Keyword"), None);
    assert_eq!(label_counts(&back).get("Concept"), Some(&1));

    // Every edge type the export carries comes back.
    for conn in ["DEPENDS_ON", "CHILD_OF", "EMBEDS", "TAGGED", "HAS_IMAGE"] {
        assert!(
            edge_types(&back).contains(&conn.to_string()),
            "{conn} did not survive: {:?}",
            edge_types(&back)
        );
    }
    // `HAS_KEYWORD` and `RELATED_TO` are the declaration-carried types, lost
    // with `vault.yaml`; `RELATED_TO` comes back as the built-in ladder's
    // `RELATED` from the same heading.
    assert!(!edge_types(&back).contains(&"HAS_KEYWORD".to_string()));
}

fn label_counts(graph: &DirGraph) -> BTreeMap<String, usize> {
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    for idx in graph.graph.node_indices() {
        if let Some(view) = graph.node_view(idx) {
            *out.entry(view.node_type_str(&graph.interner).to_string())
                .or_default() += 1;
        }
    }
    out
}

/// A vault that derives sections and chunks exports exactly the notes it
/// exported without them: a derived node carrying a `file_path` would have
/// been written out as a note of its own (VAULT.md §7.1, §10.1).
#[test]
fn deriving_structure_writes_no_extra_files() {
    let source = golden_vault();
    let plain = crate::okf::build(&source, &BuildOptions::for_dialect(Dialect::Obsidian))
        .unwrap()
        .graph;
    let structured = crate::okf::build(&source, &structure_options())
        .unwrap()
        .graph;
    assert!(
        structured.has_node_type("Section") && structured.has_node_type("Chunk"),
        "the fixture has to derive something for this to mean anything"
    );
    let plain_out = tempfile::tempdir().unwrap();
    let structured_out = tempfile::tempdir().unwrap();
    let plain_report = export_to(&plain, plain_out.path());
    let structured_report = export_to(&structured, structured_out.path());
    assert_eq!(tree(structured_out.path()), tree(plain_out.path()));
    assert_eq!(
        structured_report.edge_properties_dropped, plain_report.edge_properties_dropped,
        "no derived node is a file, so none of them can move this number"
    );
}

/// The fidelity number counts what an export **loses**, and an edge whose
/// target the export does not write is not written at all — the body's own
/// `[[note#Heading]]` is, and the next import re-derives the edge and its
/// properties from that prose (VAULT.md §10.9 loss 1).
///
/// The count ran *before* the target lookup, so every such edge was counted as
/// loss. Sections make it visible — a retargeted link lands on a node that is
/// never a file — but the same over-count applied to every `HAS_IMAGE` and
/// every hub edge before them.
#[test]
fn an_edge_to_a_target_the_export_never_writes_is_not_counted_as_loss() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            ("a.md", "## Deep\n\n[[b#Heading]]\n"),
            ("b.md", "# Heading\n\nprose\n"),
        ],
    );
    let out = tempfile::tempdir().unwrap();
    let plain = build_vault(source.path());
    assert_eq!(
        export_to(&plain, out.path()).edge_properties_dropped,
        2,
        "without sections the link ends on `b`, which is written"
    );

    let structured = crate::okf::build(source.path(), &structure_options())
        .unwrap()
        .graph;
    let structured_out = tempfile::tempdir().unwrap();
    assert_eq!(
        export_to(&structured, structured_out.path()).edge_properties_dropped,
        0,
        "with sections it ends on `b#Heading`, which is not"
    );
}

/// Sections and chunks with the spec's own defaults.
fn structure_options() -> BuildOptions {
    use crate::okf::structure::profile::{ChunkRule, SectionRule, StructureProfile};
    let mut opts = BuildOptions::for_dialect(Dialect::Obsidian);
    opts.profile.structure = Some(StructureProfile {
        sections: Some(SectionRule {
            label: "Section".to_string(),
            edge: "HAS_SECTION".to_string(),
            parent: "PARENT_SECTION".to_string(),
            next: "NEXT_SECTION".to_string(),
        }),
        chunks: Some(ChunkRule {
            label: "Chunk".to_string(),
            edge: "HAS_CHUNK".to_string(),
            next: "NEXT_CHUNK".to_string(),
            max_words: 650,
            max_chars: 6000,
        }),
        ..Default::default()
    });
    opts
}

// ── declared edge tables (VAULT.md §7.3, §10.6) ─────────────────────────────

/// A vault that declares one edge table: the `structure.tables:` rule that
/// reads it in and the `export.edge_tables:` entry that writes it back.
const EDGE_TABLE_VAULT: &[(&str, &str)] = &[
    (
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\n\
         structure:\n  tables:\n    \
         - {under_heading: '^Worked on by$', edge: WORKED_ON_BY, edges: true}\n\
         export:\n  edge_tables:\n    WORKED_ON_BY: Worked on by\n",
    ),
    (
        "Note/paper.md",
        "# Paper\n\n## Worked on by\n\n\
         | person | role |\n|---|---|\n\
         | [[alice\\|Alice A]] | author |\n| [[nobody]] | reviewer |\n",
    ),
    ("Note/alice.md", "Alice's own note.\n"),
];

fn build_with_config(dir: &Path) -> Arc<DirGraph> {
    crate::okf::build(dir, &BuildOptions::for_dialect(Dialect::Obsidian))
        .unwrap()
        .graph
}

/// The whole point (§10.6): the properties a frontmatter list cannot hold are
/// written as the table the vault declared, and the type is *not* also written
/// as a key — writing both would make the edge twice on the next import.
#[test]
fn a_declared_type_is_written_as_a_table_and_not_as_a_frontmatter_key() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), EDGE_TABLE_VAULT);
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();

    let paper = read(out.path(), "Note/paper.md");
    assert!(
        paper.contains(
            "## Worked on by\n\n\
             | person | role |\n| --- | --- |\n\
             | [[alice\\|Alice A]] | author |\n| [[nobody]] | reviewer |\n"
        ),
        "{paper}"
    );
    assert!(!paper.contains("worked_on_by:"), "{paper}");
    assert_eq!(report.warnings, Vec::<String>::new());
    // The three left are the cells' *own* prose links (`LINKS_TO` carrying
    // `section`, and `label` on the one with display text): a row rule reads a
    // cell, it never swallows it, so the body states those links as well — and
    // the body travels verbatim, which is why they come back anyway (§10.9).
    assert_eq!(
        report.edge_properties_dropped, 3,
        "none of the six WORKED_ON_BY properties is dropped; they are written"
    );
}

/// The same vault without the declaration: the type's targets go to a
/// frontmatter list and every property it carried is counted as loss 1.
#[test]
fn an_undeclared_type_keeps_todays_frontmatter_list_and_its_count() {
    let source = tempfile::tempdir().unwrap();
    let mut files = EDGE_TABLE_VAULT.to_vec();
    files[0] = (
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\n\
         structure:\n  tables:\n    \
         - {under_heading: '^Worked on by$', edge: WORKED_ON_BY, edges: true}\n",
    );
    write_vault(source.path(), &files);
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();

    let paper = read(out.path(), "Note/paper.md");
    assert!(paper.contains("worked_on_by:"), "{paper}");
    assert!(
        report.edge_properties_dropped >= 6,
        "`section`, `row` and `role` on each of the two rows: {}",
        report.edge_properties_dropped
    );
}

/// `ExportOptions::edge_tables` is how a graph that never was a vault declares
/// one — and how a caller overrides the heading the vault named.
#[test]
fn the_callers_declaration_adds_a_type_and_overrides_a_heading() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), EDGE_TABLE_VAULT);
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            edge_tables: BTreeMap::from([("WORKED_ON_BY".to_string(), "Contributors".to_string())]),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let paper = read(out.path(), "Note/paper.md");
    assert!(
        paper.contains("## Contributors\n\n| target | role |"),
        "the caller's heading wins, and the vault's table is left as prose: {paper}"
    );
}

/// No export writes a `vault.yaml` (§10.9 loss 4), so a declared table the
/// source vault has no import rule for travels as prose nobody reads back. The
/// author hears it from the export rather than from a missing edge later.
#[test]
fn a_declared_table_with_no_import_rule_warns() {
    let source = tempfile::tempdir().unwrap();
    let mut files = EDGE_TABLE_VAULT.to_vec();
    files[0] = (
        ".kglite/vault.yaml",
        "kglite_vault: 1\ndefault_label: Note\n\
         export:\n  edge_tables:\n    WROTE: Worked on by\n",
    );
    write_vault(source.path(), &files);
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("`export.edge_tables.WROTE`")
                && w.contains("no `structure.tables:` rule")),
        "{:?}",
        report.warnings
    );
}

/// A type whose only sources are nodes the export does not write — every node
/// `structure:` derives — has no note to write its table in.
#[test]
fn a_type_no_exported_note_emits_warns() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            (
                ".kglite/vault.yaml",
                "kglite_vault: 1\ndefault_label: Note\n\
                 export:\n  edge_tables:\n    HAS_PARAMETER: Parameters\n",
            ),
            ("Note/a.md", "# A\n\nProse.\n"),
        ],
    );
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("no exported source note emits HAS_PARAMETER")),
        "{:?}",
        report.warnings
    );
}

/// The export is writing a different directory: one unreadable file beside the
/// graph's source is a warning, not a refusal to write a vault.
#[test]
fn an_unreadable_source_vault_yaml_warns_rather_than_failing() {
    let source = tempfile::tempdir().unwrap();
    write_vault(source.path(), &[("Note/a.md", "Prose.\n")]);
    let broken = tempfile::tempdir().unwrap();
    write_vault(
        broken.path(),
        &[(".kglite/vault.yaml", "kglite_vault: 1\nnonsense: true\n")],
    );
    let graph = build_vault(source.path());
    let out = tempfile::tempdir().unwrap();
    let report = export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(broken.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.files_written, 1);
    assert!(
        report.warnings.iter().any(|w| w.contains("unknown key")),
        "{:?}",
        report.warnings
    );
}

/// The body states an edge of a declared type *outside* the table — a link
/// under a heading the §5.3 ladder types. It keeps its properties in that
/// prose, so a row for it would make a second edge on the next import, exactly
/// as a frontmatter list would (§10.6).
///
/// The type is one the **built-in** ladder gives, because that is the ladder
/// the export re-reads the body with: a `heading_edges:` retyping lives in
/// `vault.yaml`, which no export writes, and §10.9 loss 4 already says such an
/// edge comes back beside the ladder's own.
#[test]
fn an_edge_the_prose_states_elsewhere_gets_no_row() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            (
                ".kglite/vault.yaml",
                "kglite_vault: 1\ndefault_label: Note\n\
                 structure:\n  tables:\n    \
                 - {under_heading: '^Worked on by$', edge: RELATED, edges: true}\n\
                 export:\n  edge_tables:\n    RELATED: Worked on by\n",
            ),
            (
                "Note/paper.md",
                "# Paper\n\n## Worked on by\n\n\
                 | person | role |\n|---|---|\n| [[alice]] | author |\n\n\
                 ## Related\n\n[[bob]] is related.\n",
            ),
            ("Note/alice.md", "Alice.\n"),
            ("Note/bob.md", "Bob.\n"),
        ],
    );
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let paper = read(out.path(), "Note/paper.md");
    assert!(paper.contains("| [[alice]] | author |"), "{paper}");
    assert!(
        !paper.contains("| [[bob]]"),
        "bob is stated in prose, and that prose travels: {paper}"
    );
    assert!(paper.contains("[[bob]] is related."), "{paper}");
}

/// …and the exporter's **own** table is not the body stating anything. Under a
/// heading the ladder types, every cell is a prose link of the declared type
/// too — read as "the body already says this", every row would drop itself and
/// the table would empty on the first export, losing the properties the whole
/// feature exists to keep.
#[test]
fn the_tables_own_cells_are_not_read_as_the_body_stating_the_edge() {
    let source = tempfile::tempdir().unwrap();
    write_vault(
        source.path(),
        &[
            (
                ".kglite/vault.yaml",
                "kglite_vault: 1\ndefault_label: Note\n\
                 structure:\n  tables:\n    \
                 - {under_heading: '^Related$', edge: RELATED, edges: true}\n\
                 export:\n  edge_tables:\n    RELATED: Related\n",
            ),
            (
                "Note/paper.md",
                "# Paper\n\n## Related\n\n\
                 | person | role |\n|---|---|\n| [[alice]] | author |\n",
            ),
            ("Note/alice.md", "Alice.\n"),
        ],
    );
    let graph = build_with_config(source.path());
    let out = tempfile::tempdir().unwrap();
    export(
        &graph,
        out.path(),
        &ExportOptions {
            source_root: Some(source.path().to_path_buf()),
            ..ExportOptions::default()
        },
    )
    .unwrap();
    let paper = read(out.path(), "Note/paper.md");
    assert!(paper.contains("| [[alice]] | author |"), "{paper}");
}
