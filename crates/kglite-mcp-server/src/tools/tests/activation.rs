//! Activation-summary and revision-set build tests.

use super::*;

#[test]
fn activation_summary_reports_node_types_or_none() {
    let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    assert!(
        gs.activation_summary().is_none(),
        "no active graph → terse activation (no mini-map)"
    );
    let dir = std::env::temp_dir().join(format!("kgl_actsum_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join("m.py"),
        "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n",
    )
    .unwrap();
    gs.build_workspace_graph(&dir, None)
        .expect("build workspace graph");
    let summary = gs
        .activation_summary()
        .expect("summary present once a graph is active");
    assert!(summary.contains("Function"), "names node types: {summary}");
    assert!(
        summary.contains("graph_overview()"),
        "steers to the graph: {summary}"
    );
    assert!(
        summary.contains("search your tool registry"),
        "carries the lazy-discovery escape hatch: {summary}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn revision_build_swaps_slot_and_records_revisions() {
    let dir = std::env::temp_dir().join(format!("kgl_slot_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let (s1, s2) = ("r1".to_string(), "r2".to_string());
    let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let revs = vec![s1.clone(), s2.clone()];
    gs.build_workspace_graph(&dir, Some(&revs))
        .expect("multi-rev build");
    // The slot is active with nodes.
    let (nodes, _edges) = gs.schema().expect("schema after multi-rev build");
    assert!(nodes > 0, "multi-rev graph should have nodes");
    // `bar` exists only in the second rev → its `revs` list is a subset.
    // `foo` exists in both. Assert the rev list props landed on the merged
    // graph (the B.2b merge stamps `revs` on every node).
    let has_revs_prop = gs.has_property("Function", "revs");
    assert!(
        has_revs_prop,
        "merged multi-rev Function nodes should carry a `revs` list prop"
    );
    // The active slot records the resolved rev-set for the identity surfaces.
    let attrs = gs.with_active(|a| a.identity_attrs());
    assert!(
        attrs.contains(&format!("revs=\"{},{}\"", s1, s2)),
        "identity header should name the loaded revs; got: {attrs}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn activation_summary_teaches_rev_scoping_for_multi_rev() {
    let dir = std::env::temp_dir().join(format!("kgl_actsumrev_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let (s1, s2) = ("r1".to_string(), "r2".to_string());
    let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    gs.build_workspace_graph(&dir, Some(&[s1.clone(), s2.clone()]))
        .expect("multi-rev build");
    let summary = gs.activation_summary().expect("summary present");
    // Still carries the base mini-map + discovery hatch.
    assert!(summary.contains("Function"), "names node types: {summary}");
    // Multi-rev steer: names the revs, warns about over-count, teaches the
    // scoping idiom + rev_diff (matching the describe() provenance text).
    assert!(
        summary.contains("Multi-rev graph spanning 2"),
        "names the rev span: {summary}"
    );
    assert!(
        summary.contains("IN n.revs"),
        "teaches the `WHERE '<rev>' IN n.revs` scoping idiom: {summary}"
    );
    assert!(
        summary.contains("rev_diff"),
        "points at CALL rev_diff for deltas: {summary}"
    );
    // The newest rev (HEAD-equivalent, last in the list) is surfaced for
    // head-only scoping.
    assert!(
        summary.contains(&format!("'{s2}' IN n.revs")),
        "surfaces the newest rev for head-only scoping: {summary}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn single_rev_build_carries_no_revs_attr_or_steer() {
    // The plain build path leaves `revs = None`, so neither the header attr
    // nor the multi-rev steer appears (no regression for single-rev graphs).
    let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(test_hooks()));
    let dir = std::env::temp_dir().join(format!("kgl_single_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(dir.join("m.py"), "def foo():\n    return 1\n").unwrap();
    gs.build_workspace_graph(&dir, None)
        .expect("single-rev build");
    let attrs = gs.with_active(|a| a.identity_attrs());
    assert!(
        !attrs.contains("revs="),
        "no revs attr for single-rev: {attrs}"
    );
    let summary = gs.activation_summary().expect("summary");
    assert!(
        !summary.contains("Multi-rev graph"),
        "no multi-rev steer for single-rev: {summary}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
