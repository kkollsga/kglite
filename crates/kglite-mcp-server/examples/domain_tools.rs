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
            move |args| match graph_state.schema() {
                Some((nodes, edges)) if args.verbose => {
                    format!("active collection: {nodes} nodes and {edges} edges")
                }
                Some((nodes, _)) => format!("active collection: {nodes} nodes"),
                None => "no active collection".to_string(),
            },
        )
    });

    run_with_extensions(std::env::args_os(), extensions)
}
