use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

pub(super) fn check_target(graph: &DirGraph) -> Result<(), String> {
    if graph.graph.is_mapped() || graph.graph.is_disk() {
        return Err(
            "load_rdf currently supports the in-memory (Default) backend only; \
             mapped/disk graphs are not yet supported"
                .to_owned(),
        );
    }
    if graph.graph.plain_memory_digraph().is_none()
        || graph.graph.node_count() != 0
        || graph.graph.edge_count() != 0
        || graph.version() != 0
        || graph.checkpoint_lsn != 0
        || graph.cdc_enabled()
        || graph.read_only
        || graph.schema_locked
        || graph.active_write_scope.is_some()
        || has_schema_state(graph)
        || has_index_state(graph)
    {
        return Err(
            "load_rdf requires a fresh empty in-memory graph without schema, indexes, \
             constraints, identity aliases or mutation capture; load into DirGraph::new() instead"
                .to_owned(),
        );
    }
    Ok(())
}

fn has_schema_state(graph: &DirGraph) -> bool {
    graph.schema_definition.is_some()
        || !graph.id_field_aliases.is_empty()
        || !graph.title_field_aliases.is_empty()
        || !graph.node_type_metadata.is_empty()
        || !graph.connection_type_metadata.is_empty()
        || !graph.type_schemas.is_empty()
        || !graph.property_shapes.is_empty()
        || !graph.ontology.is_empty()
        || !graph.list_unique_constraints().is_empty()
        || !graph.ddl_not_null_constraints.is_empty()
        || !graph.ddl_property_type_constraints.is_empty()
        || !graph.rel_ddl_not_null_constraints.is_empty()
        || !graph.rel_ddl_property_type_constraints.is_empty()
}

fn has_index_state(graph: &DirGraph) -> bool {
    !graph.type_indices.is_empty()
        || !graph.id_indices.is_empty()
        || !graph.property_indices.is_empty()
        || !graph.composite_indices.is_empty()
        || !graph.range_indices.is_empty()
        || !graph.property_index_keys.is_empty()
        || !graph.composite_index_keys.is_empty()
        || !graph.range_index_keys.is_empty()
        || !graph.text_indexes.is_empty()
        || !graph.embeddings.is_empty()
}
