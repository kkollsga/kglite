use kglite_mcp_server::{run_with_extensions, ServerExtensions};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Default, Deserialize, JsonSchema)]
struct SummaryArgs {
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let extensions = ServerExtensions::new().with_domain_tools(|registry| {
        let graph_state = registry.graph_state().clone();
        registry.register_typed_tool::<SummaryArgs, _>(
            "collection_summary",
            "Summarise the active domain collection.",
            move |args| match graph_state.with_context(|context| {
                let schema = kglite::api::introspection::compute_schema(context.graph().dir());
                let root = context.root().map_or_else(
                    || "in-memory".to_string(),
                    |path| path.display().to_string(),
                );
                (schema.node_count, schema.edge_count, root)
            }) {
                Some((nodes, edges, root)) if args.verbose => {
                    format!("active collection at {root}: {nodes} nodes and {edges} edges")
                }
                Some((nodes, _, root)) => format!("active collection at {root}: {nodes} nodes"),
                None => "no active collection".to_string(),
            },
        )
    });

    run_with_extensions(std::env::args_os(), extensions)
}
