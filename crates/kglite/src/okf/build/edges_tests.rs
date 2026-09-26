//! What a written link becomes: the edge properties it carries, the heading
//! map and typed frontmatter keys that name its type, and the rules `loose`
//! keeps off.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use crate::okf::build::build;
use crate::okf::build::tests_support::{
    count_label, edges_of, labels_by_id, provisional_count, vault_build, vault_build_with, write,
};
use crate::okf::model::{BuildOptions, TAG_LABEL};
use std::collections::BTreeSet;
use tempfile::tempdir;

#[test]
fn heading_edges_beat_the_built_in_ladder_whatever_the_casing() {
    let dir = tempdir().unwrap();
    write(dir.path(), "b.md", "leaf");
    write(
        dir.path(),
        "a.md",
        "## Related Topics\n\n[[b]]\n\n## References\n\n[[b]]",
    );
    let plain = vault_build(dir.path());
    let types: BTreeSet<String> = plain
        .report
        .edges_by_type
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert!(
        types.contains("RELATED") && types.contains("REFERENCES"),
        "the built-in ladder types both sections: {types:?}"
    );

    let declared = vault_build_with(dir.path(), |p| {
        // Declared in a different casing than the heading is written in.
        p.heading_edges
            .insert("related topics".to_string(), "RELATED_TO".to_string());
    });
    assert_eq!(
        declared.report.edges_by_type.get("RELATED_TO"),
        Some(&1),
        "the map wins over the ladder"
    );
    assert_eq!(declared.report.edges_by_type.get("RELATED"), None);
    assert_eq!(
        declared.report.edges_by_type.get("REFERENCES"),
        Some(&1),
        "and a heading the map does not name keeps its ladder rung"
    );
}

#[test]
fn vault_body_link_edges_carry_section_and_anchor() {
    let dir = tempdir().unwrap();
    write(dir.path(), "b.md", "leaf");
    write(
        dir.path(),
        "a.md",
        "First [[b]] is section-less.\n\n## Deep dive\n\nThen [[b#Goals]].",
    );
    let out = vault_build(dir.path());
    assert_eq!(
        edges_of(&out.graph),
        vec![
            ("a".into(), "LINKS_TO".into(), "b".into(), vec![]),
            (
                "a".into(),
                "LINKS_TO".into(),
                "b".into(),
                vec![
                    ("anchor".to_string(), "Goals".to_string()),
                    ("section".to_string(), "Deep dive".to_string()),
                ]
            ),
        ],
        "two links differing only in their section are two edges"
    );
}

/// Two links differing only in `section` survive *another* group of the
/// same connection type being emitted first. Re-detecting the initial-load
/// regime per call made that first group register `LINKS_TO`, which flipped
/// every later group into merging and folded these two onto one edge — so
/// the graph depended on which endpoint labels happened to sort first.
#[test]
fn vault_parallel_link_edges_survive_an_earlier_group_of_the_same_type() {
    let dir = tempdir().unwrap();
    write(dir.path(), "b.md", "leaf");
    write(dir.path(), "zzz/a.md", "[[b]]\n\n## Sec\n\n[[b]] again");
    write(dir.path(), "sub/c.md", "see [[b]]");
    let out = vault_build(dir.path());
    assert_eq!(
        out.report.edges_by_type.get("LINKS_TO"),
        Some(&3),
        "the report counts three rows"
    );
    assert_eq!(
        edges_of(&out.graph)
            .iter()
            .filter(|(_, c, _, _)| c == "LINKS_TO")
            .count(),
        3,
        "and the graph holds three edges"
    );
}

/// Two source labels writing one connection type, each with two links to the
/// same note: every (type, source label) group owns its rows, so the second
/// label's parallel links are not folded onto one edge.
#[test]
fn vault_a_second_source_label_keeps_its_parallel_links() {
    let dir = tempdir().unwrap();
    write(dir.path(), "b.md", "leaf");
    write(
        dir.path(),
        "a.md",
        "---\ntype: Project\n---\n[[b]]\n\n## Sec\n\n[[b]] again",
    );
    write(
        dir.path(),
        "c.md",
        "---\ntype: Person\n---\n[[b]]\n\n## Other\n\n[[b]] again",
    );
    let out = vault_build(dir.path());
    let labels = labels_by_id(&out.graph);
    assert_eq!(
        (labels["a"].as_str(), labels["c"].as_str()),
        ("Project", "Person"),
        "the two linking notes carry different labels: {labels:?}"
    );
    let links: Vec<String> = edges_of(&out.graph)
        .into_iter()
        .filter(|(_, c, _, _)| c == "LINKS_TO")
        .map(|(s, _, _, _)| s)
        .collect();
    assert_eq!(
        links,
        vec!["a", "a", "c", "c"],
        "one edge per link, per label"
    );
}

#[test]
fn vault_embed_of_a_note_becomes_an_edge() {
    let dir = tempdir().unwrap();
    write(dir.path(), "b.md", "leaf");
    write(dir.path(), "a.md", "![[b]] and ![[diagram.png]]");
    let out = vault_build(dir.path());
    assert_eq!(
        edges_of(&out.graph)
            .into_iter()
            .map(|(s, c, t, _)| (s, c, t))
            .collect::<Vec<_>>(),
        vec![
            ("a".into(), "EMBEDS".into(), "b".into()),
            ("a".into(), "HAS_IMAGE".into(), "diagram.png".into()),
        ],
        "an image embed is an attachment (§6), not a link"
    );
    assert_eq!(
        out.report.dangling, 0,
        "and it mints no *link* stub — the absent file is a missing \
         attachment instead"
    );
    assert_eq!(out.report.missing_attachments, 1);
}

#[test]
fn vault_wikilink_valued_frontmatter_keys_become_typed_edges() {
    let dir = tempdir().unwrap();
    write(dir.path(), "x.md", "leaf");
    write(dir.path(), "y.md", "leaf");
    write(
        dir.path(),
        "a.md",
        "---\nsee_also: \"[[x]]\"\ndepends_on:\n- \"[[x]]\"\n- \"[[y]]\"\nreviewers:\n- \"[[x]]\"\n- ada\n---\nbody",
    );
    let out = vault_build(dir.path());
    assert_eq!(
        edges_of(&out.graph),
        vec![
            ("a".into(), "DEPENDS_ON".into(), "x".into(), vec![]),
            ("a".into(), "DEPENDS_ON".into(), "y".into(), vec![]),
            ("a".into(), "SEE_ALSO".into(), "x".into(), vec![]),
        ]
    );
    let n = out
        .graph
        .graph
        .node_indices()
        .find(|&n| {
            out.graph.node_view(n).map(|nd| nd.id().into_owned()) == Some(Value::String("a".into()))
        })
        .unwrap();
    let prop =
        |k: &str| GraphRead::get_node_property(&out.graph.graph, n, InternedKey::from_str(k));
    assert_eq!(prop("see_also"), None, "an edge key is not also a property");
    assert_eq!(prop("depends_on"), None);
    assert_eq!(
        prop("reviewers"),
        Some(Value::List(vec![
            Value::String("[[x]]".into()),
            Value::String("ada".into()),
        ])),
        "a list mixing wikilinks with plain strings stays a property"
    );
}

#[test]
fn loose_keeps_every_vault_link_rule_off() {
    let dir = tempdir().unwrap();
    write(dir.path(), "x.md", "---\ntype: Note\n---\nleaf");
    write(
        dir.path(),
        "a.md",
        "---\ntype: Note\ndepends_on: \"[[x]]\"\naliases:\n- Ex\n---\n## Sec\n\n#tag and [[x]] and ![[x]] and [[Ex]]",
    );
    let out = build(
        dir.path(),
        &BuildOptions::for_dialect(crate::okf::Dialect::Loose),
    )
    .unwrap();
    assert_eq!(
        edges_of(&out.graph),
        vec![
            ("a".into(), "LINKS_TO".into(), "Ex".into(), vec![]),
            ("a".into(), "LINKS_TO".into(), "x".into(), vec![]),
        ],
        "two plain wikilinks: no section, no EMBEDS, no frontmatter edge"
    );
    assert_eq!(
        provisional_count(&out.graph),
        1,
        "`[[Ex]]` dangles — the alias rung is a vault rule"
    );
    assert_eq!(count_label(&out.graph, TAG_LABEL), 0);
    assert_eq!(
        GraphRead::get_node_property(
            &out.graph.graph,
            out.graph
                .graph
                .node_indices()
                .find(|&n| out.graph.node_view(n).map(|nd| nd.id().into_owned())
                    == Some(Value::String("a".into())))
                .unwrap(),
            InternedKey::from_str("depends_on")
        ),
        Some(Value::String("[[x]]".into())),
        "the key stays an ordinary property outside a vault"
    );
}
