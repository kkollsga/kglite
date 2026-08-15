//! The single source of truth for the procedure surface.
//!
//! Three consumers read this table — `valid_yield_columns` (YIELD
//! validation + bare-CALL expansion), the `list_procedures` procedure, and
//! the `SHOW PROCEDURES` statement. It exists because the first two were
//! previously two hand-maintained lists that had already drifted apart
//! (`list_procedures` advertised `db.labels` as yielding `name`; the
//! validator said `label` — the validator was right). A procedure that is
//! not in this table does not exist to any of the three.
//!
//! `db.checkpoint()` and the `dbms.*` verbs are deliberately absent: they
//! are Bolt-server intercepts reporting server state the engine does not
//! hold (see `kglite-bolt-server::backend`), so listing them here would
//! advertise calls that fail in every non-Bolt binding.

/// One procedure: canonical spelling, accepted aliases, a one-line
/// description, and the declared YIELD columns in declared order.
pub(super) struct ProcedureSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub columns: &'static [&'static str],
}

/// Match key: procedure names arrive lowercased from the parser, and the
/// canonical spellings keep display case (`db.relationshipTypes`), so the
/// lookup is case-insensitive.
pub(super) fn find_procedure(name: &str) -> Option<&'static ProcedureSpec> {
    PROCEDURES.iter().find(|spec| {
        spec.name.eq_ignore_ascii_case(name)
            || spec
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
    })
}

pub(super) const PROCEDURES: &[ProcedureSpec] = &[
    // ── graph algorithms ──
    ProcedureSpec {
        name: "pagerank",
        aliases: &[],
        description: "Compute PageRank centrality for all nodes",
        columns: &["node", "score"],
    },
    ProcedureSpec {
        name: "betweenness",
        aliases: &["betweenness_centrality"],
        description: "Compute betweenness centrality for all nodes",
        columns: &["node", "score"],
    },
    ProcedureSpec {
        name: "degree",
        aliases: &["degree_centrality"],
        description: "Compute degree centrality for all nodes",
        columns: &["node", "score"],
    },
    ProcedureSpec {
        name: "closeness",
        aliases: &["closeness_centrality"],
        description: "Compute closeness centrality for all nodes",
        columns: &["node", "score"],
    },
    ProcedureSpec {
        name: "louvain",
        aliases: &["louvain_communities"],
        description: "Detect communities using multilevel Louvain (hierarchical). YIELD optional 'level' for the community hierarchy. Params: {resolution, weight_property, connection_types}",
        columns: &["node", "community", "level"],
    },
    ProcedureSpec {
        name: "leiden",
        aliases: &["leiden_communities"],
        description: "Detect communities using Leiden (multilevel, well-connected communities). YIELD optional 'level' for the hierarchy. Params: {resolution, weight_property, connection_types}",
        columns: &["node", "community", "level"],
    },
    ProcedureSpec {
        name: "label_propagation",
        aliases: &[],
        description: "Detect communities using label propagation",
        columns: &["node", "community"],
    },
    ProcedureSpec {
        name: "connected_components",
        aliases: &["weakly_connected_components"],
        description: "Find weakly connected components. Optional {node_type, relationship} scoping to a subgraph.",
        columns: &["node", "component"],
    },
    ProcedureSpec {
        name: "k_core",
        aliases: &["coreness"],
        description: "k-core decomposition (coreness per node). Optional {node_type, relationship} scoping. Filter WHERE coreness >= k for the k-core.",
        columns: &["node", "coreness"],
    },
    ProcedureSpec {
        name: "ready_set",
        aliases: &["dependency_frontier"],
        description: "Dependency frontier: nodes whose {edge} prerequisites all satisfy the `done` predicate — the next actionable work items. Params: {node_type, edge, done}",
        columns: &["node", "dependency_count"],
    },
    ProcedureSpec {
        name: "clustering_coefficient",
        aliases: &["local_clustering_coefficient"],
        description: "Local clustering coefficient per node (how interconnected its neighbours are). Optional {node_type, relationship} scoping.",
        columns: &["node", "coefficient"],
    },
    ProcedureSpec {
        name: "triangle_count",
        aliases: &["transitivity"],
        description: "Global triangle count + transitivity (global clustering coefficient) for the whole graph. Single aggregate row. Optional {node_type, relationship} scoping. (Alias: transitivity.)",
        columns: &["triangles", "transitivity"],
    },
    ProcedureSpec {
        name: "eccentricity",
        aliases: &[],
        description: "Per-node eccentricity (longest shortest path to any node in its component). All-pairs BFS, capped at 20k scoped nodes — narrow with {node_type, relationship}.",
        columns: &["node", "eccentricity"],
    },
    ProcedureSpec {
        name: "diameter",
        aliases: &[],
        description: "Graph diameter (max eccentricity). Single aggregate row. Same all-pairs cost + 20k-node cap as eccentricity.",
        columns: &["diameter"],
    },
    ProcedureSpec {
        name: "cluster",
        aliases: &[],
        description: "Cluster nodes by spatial location or numeric properties (DBSCAN/K-means). Reads from preceding MATCH.",
        columns: &["node", "cluster"],
    },
    // ── rule / validation procedures ──
    ProcedureSpec {
        name: "orphan_node",
        aliases: &[],
        description: "Rule: nodes of {type} with zero matching edges (default: any edge, both directions). Optional: link_type='X' restricts to that connection type; direction='in'|'out'|'both'.",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "self_loop",
        aliases: &[],
        description: "Rule: nodes of {type} with a self-loop via {edge}",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "cycle_2step",
        aliases: &[],
        description: "Rule: a-{edge}->b-{edge}->a pairs where both nodes are of {type}",
        columns: &["node_a", "node_b"],
    },
    ProcedureSpec {
        name: "missing_required_edge",
        aliases: &[],
        description: "Rule: nodes of {type} with no outgoing edge of {edge} (direction-validated)",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "missing_inbound_edge",
        aliases: &[],
        description: "Rule: nodes of {type} with no incoming edge of {edge} (direction-validated)",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "duplicate_title",
        aliases: &[],
        description: "Rule: nodes of {type} whose title is shared with another node of the same type",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "duplicate_id",
        aliases: &[],
        description: "Rule: nodes of {type} whose id is shared with another node of the same type",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "null_property",
        aliases: &[],
        description: "Rule: nodes of {type} where {property} is missing, null, or empty",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "outline",
        aliases: &[],
        description: "Projection: BFS spanning tree from node id {root} along {edge} — the tree structure (render with kglite.outline)",
        columns: &["node", "depth", "parent_id"],
    },
    ProcedureSpec {
        name: "inverse_violation",
        aliases: &[],
        description: "Rule: (a)-[rel_a]->(b) without a matching (b)-[rel_b]->(a)",
        columns: &["a", "b"],
    },
    ProcedureSpec {
        name: "transitivity_violation",
        aliases: &[],
        description: "Rule: (a)->(b)->(c) chains under {rel} where the direct (a)->(c) edge is absent",
        columns: &["a", "b", "c"],
    },
    ProcedureSpec {
        name: "cardinality_violation",
        aliases: &[],
        description: "Rule: nodes of {type} whose outgoing-{edge} count is outside [min, max]",
        columns: &["node", "count"],
    },
    ProcedureSpec {
        name: "type_domain_violation",
        aliases: &[],
        description: "Rule: edges of {edge} whose source node is not of {expected_source} type",
        columns: &["source", "target"],
    },
    ProcedureSpec {
        name: "type_range_violation",
        aliases: &[],
        description: "Rule: edges of {edge} whose target node is not of {expected_target} type",
        columns: &["source", "target"],
    },
    ProcedureSpec {
        name: "parallel_edges",
        aliases: &[],
        description: "Rule: (a, b) pairs connected by more than one edge of {edge}",
        columns: &["a", "b", "count"],
    },
    ProcedureSpec {
        name: "kg_knn",
        aliases: &[],
        description: "Spatial: k nearest nodes of {target_type} to ({lat}, {lon})",
        columns: &["node", "distance_m"],
    },
    // ── code-graph analysis ──
    ProcedureSpec {
        name: "affected_tests",
        aliases: &[],
        description: "Code graphs: test files reachable from changed files via inbound IMPORTS edges. Params: {files, max_depth}",
        columns: &["test_file", "depth"],
    },
    ProcedureSpec {
        name: "rev_diff",
        aliases: &[],
        description: "Multi-rev code graphs: added/removed/changed code entities between two revs {from, to}. Reads revs/rev_fp list props (stamped by a multi-rev code-graph build, e.g. codingest --revs). Optional {node_type} scoping.",
        columns: &["bucket", "type", "qualified_name", "name", "file", "line"],
    },
    ProcedureSpec {
        name: "dead_code",
        aliases: &[],
        description: "Functions with no inbound use edge (CALLS / REFERENCES_FN / HANDLES / IMPLEMENTED_BY / DECORATES); excludes tests, dunder and main (pass exclude_public to also drop pub/exported, include_tests to keep tests)",
        columns: &["node"],
    },
    ProcedureSpec {
        name: "refresh_stats",
        aliases: &[],
        description: "Recompute the label-pair edge-count cache; one row per (src_type, edge_type, tgt_type) with its fresh count. Diagnostic for what the planner thinks the schema looks like.",
        columns: &["src_type", "edge_type", "tgt_type", "count"],
    },
    // ── meta ──
    ProcedureSpec {
        name: "list_procedures",
        aliases: &[],
        description: "List all available procedures",
        columns: &["name", "description", "yield_columns"],
    },
    // ── Neo4j-compatible schema introspection (db.*) ──
    ProcedureSpec {
        name: "db.labels",
        aliases: &[],
        description: "All node-type names ('labels') in the graph, sorted",
        columns: &["label"],
    },
    ProcedureSpec {
        name: "db.relationshipTypes",
        aliases: &[],
        description: "All connection-type names ('relationship types') in the graph, sorted",
        columns: &["relationshipType"],
    },
    ProcedureSpec {
        name: "db.indexes",
        aliases: &[],
        description: "All indexes in the graph (equality, composite, range), sorted by name",
        columns: &["name", "type", "entityType", "labelsOrTypes", "properties", "state"],
    },
    ProcedureSpec {
        name: "db.constraints",
        aliases: &[],
        description: "All declared constraints (UNIQUENESS, NODE_KEY, NODE_PROPERTY_EXISTENCE), sorted by name",
        columns: &["name", "type", "entityType", "labelsOrTypes", "properties"],
    },
    ProcedureSpec {
        name: "db.propertyKeys",
        aliases: &[],
        description: "All property keys declared in the graph (node + relationship), sorted",
        columns: &["propertyKey"],
    },
    ProcedureSpec {
        name: "db.schema",
        aliases: &[],
        description: "One row per node type with its sorted property-name list",
        columns: &["nodeType", "properties"],
    },
    ProcedureSpec {
        name: "db.schema.visualization",
        aliases: &[],
        description: "Schema graph for visualization: one row with virtual nodes (one per label; properties name/indexes/constraints) and virtual relationships (one per observed source-label/type/target-label combination). What Neo4j Browser's schema tab renders.",
        columns: &["nodes", "relationships"],
    },
    ProcedureSpec {
        name: "db.schema.nodeTypeProperties",
        aliases: &[],
        description: "Typed node schema: one row per (label, property) with propertyTypes and mandatory — the shape Neo4j clients load their data model from. A property-less label emits one row with null propertyName.",
        columns: &["nodeType", "nodeLabels", "propertyName", "propertyTypes", "mandatory"],
    },
    ProcedureSpec {
        name: "db.schema.relTypeProperties",
        aliases: &[],
        description: "Typed relationship schema: one row per (type, property) with propertyTypes and mandatory. A property-less type emits one row with null propertyName.",
        columns: &["relType", "propertyName", "propertyTypes", "mandatory"],
    },
    ProcedureSpec {
        name: "apoc.meta.nodeTypeProperties",
        aliases: &[],
        description: "APOC-compatibility shim over db.schema.nodeTypeProperties: same typed node schema under APOC's column set (adds totalObservations/propertyObservations). One of exactly two apoc.* names KGLite answers; schema clients (G.V()) call the APOC pair first.",
        columns: &["nodeType", "nodeLabels", "propertyName", "propertyTypes", "mandatory", "propertyObservations", "totalObservations"],
    },
    ProcedureSpec {
        name: "apoc.meta.relTypeProperties",
        aliases: &[],
        description: "APOC-compatibility shim over db.schema.relTypeProperties, adding the endpoint columns (sourceNodeLabels/targetNodeLabels) the db.schema contract lacks - one row per observed (source, type, target) pairing. This is what schema-graph clients draw their edges from.",
        columns: &["relType", "sourceNodeLabels", "targetNodeLabels", "propertyName", "propertyTypes", "mandatory", "propertyObservations", "totalObservations"],
    },
    ProcedureSpec {
        name: "db.graph_stats",
        aliases: &[],
        description: "Per-graph summary: node, edge, label, and relationship-type counts. Single row.",
        columns: &["node_count", "edge_count", "label_count", "relationship_type_count"],
    },
    ProcedureSpec {
        name: "db.property_stats",
        aliases: &[],
        description: "Per-(label, property) statistics: value, null, and distinct counts. Params: {node_type, property}",
        columns: &["value_count", "null_count", "distinct_count"],
    },
    ProcedureSpec {
        name: "db.property_uniqueness",
        aliases: &[],
        description: "Uniqueness pre-flight for a (label, property): is it unique, and how many violations. Params: {node_type, property}",
        columns: &["is_unique", "violation_count", "distinct_count"],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical names and aliases must be unique across the table — a
    /// duplicate would make `find_procedure` silently answer with whichever
    /// entry comes first.
    #[test]
    fn names_and_aliases_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for spec in PROCEDURES {
            assert!(
                seen.insert(spec.name.to_ascii_lowercase()),
                "duplicate procedure name: {}",
                spec.name
            );
            for alias in spec.aliases {
                assert!(
                    seen.insert(alias.to_ascii_lowercase()),
                    "duplicate alias: {alias}"
                );
            }
        }
    }

    /// Lookup is case-insensitive and covers aliases.
    #[test]
    fn find_procedure_matches_case_insensitively() {
        assert_eq!(
            find_procedure("db.relationshiptypes").unwrap().name,
            "db.relationshipTypes"
        );
        assert_eq!(find_procedure("PAGERANK").unwrap().name, "pagerank");
        assert_eq!(
            find_procedure("betweenness_centrality").unwrap().name,
            "betweenness"
        );
        assert!(find_procedure("db.checkpoint").is_none());
        assert!(find_procedure("dbms.components").is_none());
    }

    /// Every spec declares at least one column — a zero-column procedure
    /// would make bare-CALL expansion produce an empty projection.
    #[test]
    fn every_spec_declares_columns() {
        for spec in PROCEDURES {
            assert!(
                !spec.columns.is_empty(),
                "{} declares no columns",
                spec.name
            );
            assert!(
                !spec.description.is_empty(),
                "{} has no description",
                spec.name
            );
        }
    }
}
