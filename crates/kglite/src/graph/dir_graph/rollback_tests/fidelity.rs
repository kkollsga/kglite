//! Per-shape rollback fidelity

use super::*;

/// Every mid-statement-failure shape, run against every graph configuration.
///
/// One entry generates one module holding one test per fixture, so a failure
/// names both the shape and the configuration it failed in
/// (`create_nodes::columnar`).
///
/// The extra arms exist because the plain fixture cannot detect the bug class
/// this file is for. `seeded()` is never saved and never indexed, so it can
/// never trip a `journal_covers` veto — it takes the journal path
/// unconditionally, and is therefore structurally incapable of noticing a
/// journal path that is wrong for a graph that *has* been saved or indexed.
/// Every shape must hold in every configuration a real application graph can
/// be in.
macro_rules! rollback_shapes {
    ($($(#[$doc:meta])* $name:ident: $query:expr, $scope:expr;)*) => {
        $(
            $(#[$doc])*
            mod $name {
                use super::*;

                /// A fresh in-memory graph: no column stores, no user indexes.
                #[test]
                fn plain() {
                    assert_rolls_back(&mut seeded(), $query, $scope);
                }

                /// The saved-graph shape — `enable_columnar` has installed
                /// master column stores that no mutation path ever removes.
                #[test]
                fn columnar() {
                    assert_rolls_back(&mut seeded_columnar(), $query, $scope);
                }

                /// The indexed shape — one user index of each family, whose
                /// buckets the statement's writes maintain incrementally.
                #[test]
                fn indexed() {
                    assert_rolls_back(&mut seeded_indexed(), $query, $scope);
                }

                /// The **mapped** shape — the storage mode a large-graph
                /// application runs in, and a journal-path configuration
                /// since 2026-07-30. Before that flip Mapped took the clone
                /// checkpoint, so no shape here had ever been rolled back
                /// through the journal on it.
                #[test]
                fn mapped() {
                    assert_rolls_back(&mut seeded_mapped(), $query, $scope);
                }
            }
        )*
    };
}

rollback_shapes! {
    create_nodes:
        "CREATE (:Item {id: 100}), (:Item {id: 101, bad: duration({months: 2147483648})})",
        None;

    /// The first pattern (nodes + edge) commits, the second is rejected by the
    /// write whitelist — so the journal must reverse two nodes and an edge.
    create_nodes_and_edges:
        "CREATE (x:Item {id: 200})-[:LINKS {weight: 1}]->(y:Item {id: 201}), \
                (z:Blocked {id: 202})",
        Some(&["Item"]);

    create_with_secondary_labels:
        "CREATE (:Tag:Hot:Fresh {id: 300}), (:Blocked {id: 301})",
        Some(&["Tag"]);

    /// Every Item is updated, then the expression on the last SET item blows
    /// up — a multi-property SET across multiple rows.
    set_properties:
        "MATCH (n:Item) SET n.qty = n.qty + 1, n.name = 'touched', \
                             n.bad = duration({months: 2147483648})",
        None;

    /// One SET clause writing two node types: the Item write commits, then the
    /// Tag write is rejected by the whitelist mid-clause.
    set_on_second_type:
        "MATCH (n:Item), (t:Tag) WHERE n.id = 1 AND t.id = 1 \
         SET n.marker = 'x', t.marker = 'y'",
        Some(&["Item"]);

    set_label:
        "MATCH (t:Tag {id: 2}) SET t:Hot, t.bad = duration({months: 2147483648})",
        None;

    remove_property_and_label:
        "MATCH (t:Tag {id: 1}) REMOVE t.name, t:Hot \
         CREATE (:Blocked {id: 400})",
        Some(&["Tag"]);

    /// Deletes the middle Item — so its slot is a hole in the middle of the
    /// type_indices bucket, and restoring it at the end instead of in place
    /// would fail the fingerprint.
    detach_delete_one:
        "MATCH (n:Item {id: 2}) DETACH DELETE n CREATE (:Blocked {id: 500})",
        Some(&["Item"]);

    detach_delete_all:
        "MATCH (n) DETACH DELETE n CREATE (:Blocked {id: 501})",
        Some(&["Item", "Tag"]);

    delete_labelled_node:
        "MATCH (t:Tag) DETACH DELETE t CREATE (:Blocked {id: 502})",
        Some(&["Tag"]);

    delete_edge:
        "MATCH ()-[r:LINKS]->() DELETE r CREATE (:Blocked {id: 600})",
        Some(&["Item"]);

    merge_create_arm:
        "MERGE (n:Item {id: 700}) ON CREATE SET n.name = 'new' \
         CREATE (:Blocked {id: 701})",
        Some(&["Item"]);

    merge_match_arm:
        "MERGE (n:Item {id: 1}) ON MATCH SET n.name = 'seen' \
         CREATE (:Blocked {id: 702})",
        Some(&["Item"]);

    foreach:
        "FOREACH (i IN [1, 2, 3] | CREATE (:Item {id: 800 + i})) \
         CREATE (:Blocked {id: 804})",
        Some(&["Item"]);

    multi_clause_create_then_set_then_delete:
        "MATCH (n:Item {id: 3}) SET n.qty = 999 \
         CREATE (:Item {id: 900}) \
         CREATE (:Blocked {id: 901})",
        Some(&["Item"]);
}
