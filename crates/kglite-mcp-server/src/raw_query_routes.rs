use std::{collections::HashMap, sync::Arc};

use mcp_methods::server::McpServer;
use rmcp::handler::server::tool::DynCallToolHandler;
use rmcp::model::{CallToolResult, ContentBlock};

use crate::raw_stdio::RawJsonRpcRequest;

pub(crate) fn protect_query_route(
    server: &mut McpServer,
    name: &str,
    pointer: &'static [&'static str],
) {
    let Some(route) = server.tool_router_mut().map.get_mut(name) else {
        return;
    };
    let original = route.call.clone();
    let call: Arc<DynCallToolHandler<McpServer>> = Arc::new(move |context| {
        let original = original.clone();
        Box::pin(async move {
            if let Some(raw) = context
                .request_context
                .extensions
                .get::<RawJsonRpcRequest>()
            {
                if let Err(error) =
                    kglite::api::param::validate_json_query_numbers_at(&raw.source, pointer)
                {
                    return Ok(
                        CallToolResult::error(vec![ContentBlock::text(error.to_string())]).into(),
                    );
                }
                if raw.recovered {
                    return Ok(CallToolResult::error(vec![ContentBlock::text(
                        "query parameters could not be decoded losslessly",
                    )])
                    .into());
                }
            }
            original(context).await
        })
    });
    route.call = call;
}

pub(crate) const CYPHER_QUERY_POINTER: &[&str] = &["params", "arguments", "params"];
pub(crate) const TEMPLATE_QUERY_POINTER: &[&str] = &["params", "arguments"];
pub(crate) const RECIPE_QUERY_POINTER: &[&str] = &["params", "arguments", "variables"];

pub(crate) fn route_pointers(
    manifest: Option<&mcp_methods::server::Manifest>,
    has_recipes: bool,
) -> HashMap<String, &'static [&'static str]> {
    let mut routes = HashMap::from([("cypher_query".to_string(), CYPHER_QUERY_POINTER)]);
    if let Some(manifest) = manifest {
        for tool in &manifest.tools {
            if let mcp_methods::server::ToolSpec::Cypher(spec) = tool {
                routes.insert(spec.name.clone(), TEMPLATE_QUERY_POINTER);
            }
        }
    }
    if has_recipes {
        routes.insert("run_recipe_query".to_string(), RECIPE_QUERY_POINTER);
    }
    routes
}
