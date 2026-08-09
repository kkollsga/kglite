//! `graph_overview` decoration and temp-cleanup predicate tests.

use super::*;

#[test]
fn overview_bare_predicate_requires_an_argument_free_call() {
    assert!(OverviewArgs::default().is_bare());
    assert!(!OverviewArgs {
        types: Some(Vec::new()),
        ..OverviewArgs::default()
    }
    .is_bare());
    assert!(!OverviewArgs {
        connections: Some(DetailSelection::Enabled(false)),
        ..OverviewArgs::default()
    }
    .is_bare());
    assert!(!OverviewArgs {
        connections: Some(DetailSelection::Topics(Vec::new())),
        ..OverviewArgs::default()
    }
    .is_bare());
    assert!(!OverviewArgs {
        cypher: Some(DetailSelection::Enabled(false)),
        ..OverviewArgs::default()
    }
    .is_bare());
    assert!(!OverviewArgs {
        cypher: Some(DetailSelection::Topics(Vec::new())),
        ..OverviewArgs::default()
    }
    .is_bare());
}

#[test]
fn overview_decorations_render_prefix_body_and_catalog_in_order() {
    let decorations = OverviewDecorations {
        prefix: Some("operator prefix\n".to_string()),
        catalog: Some(catalog_summary()),
    };
    let body = "<active_graph/>\n<schema/>".to_string();

    assert_eq!(
        decorations.render(body.clone(), true),
        "operator prefix\n\
         <active_graph/>\n\
         <schema/>\n\
         <query-catalog recipes=\"2\" queries=\"5\" list-tool=\"list_recipe_queries\" run-tool=\"run_recipe_query\"/>"
    );
    assert_eq!(
        decorations.render(body.clone(), false),
        body,
        "focused overviews must remain byte-for-byte unchanged"
    );
    assert_eq!(
        OverviewDecorations::default().render(body.clone(), true),
        body,
        "absent prefix and catalog must preserve the legacy body"
    );
    assert_eq!(
        OverviewDecorations {
            prefix: Some("prefix only".to_string()),
            catalog: None,
        }
        .render("body".to_string(), true),
        "prefix only\nbody"
    );
    assert_eq!(
        OverviewDecorations {
            prefix: None,
            catalog: Some(catalog_summary()),
        }
        .render("body".to_string(), true),
        "body\n<query-catalog recipes=\"2\" queries=\"5\" list-tool=\"list_recipe_queries\" run-tool=\"run_recipe_query\"/>"
    );
}

#[test]
fn bare_overview_decorations_include_no_active_graph_body() {
    let decorations = OverviewDecorations {
        prefix: Some("operator prefix".to_string()),
        catalog: Some(catalog_summary()),
    };
    let rendered = decorations.render(NO_GRAPH.to_string(), true);

    assert!(rendered.starts_with("operator prefix\nNo active graph."));
    assert!(rendered.ends_with(
        "<query-catalog recipes=\"2\" queries=\"5\" list-tool=\"list_recipe_queries\" run-tool=\"run_recipe_query\"/>"
    ));
}

#[test]
fn temp_cleanup_uses_the_same_bare_predicate_as_decorations() {
    let temp = tempfile::tempdir().expect("tempdir");
    let cleanup_dir = temp.path().join("temp");
    std::fs::create_dir(&cleanup_dir).expect("cleanup dir");

    let focused_calls = [
        OverviewArgs {
            types: Some(Vec::new()),
            ..OverviewArgs::default()
        },
        OverviewArgs {
            connections: Some(DetailSelection::Enabled(false)),
            ..OverviewArgs::default()
        },
        OverviewArgs {
            cypher: Some(DetailSelection::Topics(Vec::new())),
            ..OverviewArgs::default()
        },
    ];
    for (index, args) in focused_calls.iter().enumerate() {
        let marker = cleanup_dir.join(format!("focused-{index}"));
        std::fs::write(&marker, "keep").expect("focused marker");
        assert!(!prepare_overview(args, true, Some(&cleanup_dir)));
        assert!(marker.exists(), "focused call must not clean temp files");
    }

    let retained = cleanup_dir.join("cleanup-disabled");
    std::fs::write(&retained, "keep").expect("disabled marker");
    assert!(prepare_overview(
        &OverviewArgs::default(),
        false,
        Some(&cleanup_dir)
    ));
    assert!(retained.exists(), "disabled cleanup must retain temp files");

    assert!(prepare_overview(
        &OverviewArgs::default(),
        true,
        Some(&cleanup_dir)
    ));
    assert_eq!(
        std::fs::read_dir(&cleanup_dir)
            .expect("read cleanup dir")
            .count(),
        0,
        "bare call cleans every accumulated entry"
    );
}
