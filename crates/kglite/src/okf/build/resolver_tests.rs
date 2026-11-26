//! The resolution ladder: which rung answers a written target, and what is
//! left dangling when none does.

use super::normalize_slug;
use crate::graph::storage::GraphRead;
use crate::okf::build::build;
use crate::okf::build::tests_support::{edges_of, provisional_count, vault_build, write};
use crate::okf::model::BuildOptions;
use tempfile::tempdir;

#[test]
fn vault_path_links_resolve_by_path_not_by_last_segment() {
    let dir = tempdir().unwrap();
    // Two notes whose stems slugify alike; only the full path tells them
    // apart, and the forgiving last-segment rung would answer the wrong one.
    write(dir.path(), "Docs/Guide.md", "---\nid: g1\n---\nvendor copy");
    write(dir.path(), "notes/guide.md", "my notes");
    write(dir.path(), "a.md", "See [g](notes/guide.md).");
    let opts = BuildOptions::for_dialect(crate::okf::Dialect::Obsidian);
    let out = build(dir.path(), &opts).unwrap();
    assert_eq!(out.report.dangling, 0, "a path link is not a dangling link");
    assert_eq!(provisional_count(&out.graph), 0);
    let g = &out.graph;
    let targets: Vec<String> = g
        .graph
        .edge_indices()
        .filter(|&e| {
            g.graph
                .edge_weight(e)
                .is_some_and(|w| w.connection_type_str(&g.interner) == "LINKS_TO")
        })
        .filter_map(|e| g.graph.edge_endpoints(e))
        .filter_map(|(_, t)| {
            g.node_view(t)
                .map(|nd| nd.node_type_str(&g.interner).to_string())
        })
        .collect();
    assert_eq!(
        targets,
        vec!["notes".to_string()],
        "the link lands on notes/guide.md (label `notes`), not on Docs/Guide.md"
    );
}

/// The ladder's first rungs apply to a wikilink too (VAULT.md §5.2): a
/// declared `id:` is a link target, and a `/`-bearing name is a path —
/// vault-relative first, then relative to the linking note.
#[test]
fn vault_wikilinks_resolve_by_id_and_by_path() {
    let dir = tempdir().unwrap();
    write(dir.path(), "notes/meeting.md", "---\nid: mtg-1\n---\nleaf");
    write(dir.path(), "wing/sub/target.md", "leaf");
    write(dir.path(), "wing/other.md", "leaf");
    write(
        dir.path(),
        "wing/a.md",
        "[[mtg-1]] and [[wing/sub/target]] and [[sub/target]]",
    );
    let out = vault_build(dir.path());
    assert_eq!(out.report.dangling, 0, "no rung fell through to a stub");
    let targets: Vec<String> = edges_of(&out.graph)
        .into_iter()
        .filter(|(_, c, _, _)| c == "LINKS_TO")
        .map(|(_, _, t, _)| t)
        .collect();
    assert_eq!(
        targets,
        vec!["mtg-1".to_string(), "target".to_string()],
        "both spellings reach that note, and land on one edge: they sit in \
         the same section with no anchor, so VAULT.md §5.4 makes them one \
         relationship"
    );
}

#[test]
fn vault_alias_resolves_a_link_and_is_not_a_stub() {
    let dir = tempdir().unwrap();
    write(
        dir.path(),
        "seismic.md",
        "---\naliases:\n- Seismic interpretation\n---\nleaf",
    );
    write(dir.path(), "a.md", "see [[Seismic interpretation]]");
    let out = vault_build(dir.path());
    assert_eq!(out.report.dangling, 0);
    assert_eq!(
        edges_of(&out.graph),
        vec![("a".into(), "LINKS_TO".into(), "seismic".into(), vec![])]
    );
    // A scalar `aliases:` names the one alias it spells.
    write(
        dir.path(),
        "seismic.md",
        "---\naliases: Seismic interpretation\n---\nleaf",
    );
    assert_eq!(vault_build(dir.path()).report.dangling, 0);
}

#[test]
fn vault_alias_collisions_warn() {
    let dir = tempdir().unwrap();
    write(dir.path(), "atlas.md", "leaf");
    write(
        dir.path(),
        "one.md",
        "---\naliases:\n- atlas\n---\nclaims a stem",
    );
    write(dir.path(), "two.md", "---\naliases:\n- shared\n---\nleaf");
    write(dir.path(), "three.md", "---\naliases:\n- shared\n---\nleaf");
    let r = vault_build(dir.path()).report;
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert!(
        r.warnings[0].contains("alias `atlas` on one.md") && r.warnings[0].contains("`atlas`"),
        "{}",
        r.warnings[0]
    );
    assert!(
        r.warnings[1].contains("alias `shared`")
            && r.warnings[1].contains("`three`")
            && r.warnings[1].contains("`two`"),
        "{}",
        r.warnings[1]
    );
}

#[test]
fn slug_normalization_unifies_separators() {
    assert_eq!(
        normalize_slug("Project_0-10 Shipped"),
        "project-0-10-shipped"
    );
    assert_eq!(
        normalize_slug("feedback_cypher_first"),
        "feedback-cypher-first"
    );
    assert_eq!(normalize_slug("--A__B--"), "a-b");
}

#[test]
fn resolves_slug_and_title_variants_without_dangling() {
    let dir = tempdir().unwrap();
    // file uses underscores; wikilinks use hyphen-slug and the human title.
    write(
        dir.path(),
        "feedback_cypher_first.md",
        "---\ntype: Note\ntitle: Cypher First\n---\nleaf",
    );
    write(
        dir.path(),
        "a.md",
        "---\ntype: Note\n---\nsee [[feedback-cypher-first]] and [[Cypher First]]",
    );
    // `for_dialect`, not a struct literal: the profile carries `wikilinks`
    // now, and a literal would leave it off and test nothing.
    let opts = BuildOptions::for_dialect(crate::okf::model::Dialect::Loose);
    let g = build(dir.path(), &opts).unwrap().graph;
    // both wikilinks resolve to the one file → 2 nodes, no provisional stub.
    assert_eq!(g.graph.node_indices().count(), 2);
    assert_eq!(provisional_count(&g), 0);
}

#[test]
fn genuinely_missing_target_still_dangles() {
    let dir = tempdir().unwrap();
    write(
        dir.path(),
        "a.md",
        "---\ntype: Note\n---\nsee [[truly-absent]]",
    );
    // `for_dialect`, not a struct literal: the profile carries `wikilinks`
    // now, and a literal would leave it off and test nothing.
    let opts = BuildOptions::for_dialect(crate::okf::model::Dialect::Loose);
    let g = build(dir.path(), &opts).unwrap().graph;
    assert_eq!(provisional_count(&g), 1);
}
