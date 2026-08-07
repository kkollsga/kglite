//! Fixed MCP route registration for the boot-validated recipe catalog.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use mcp_methods::server::McpServer;
use rmcp::handler::server::router::tool::ToolRoute;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::model::{CallToolResponse, Tool, ToolAnnotations};
use rmcp::ErrorData as McpError;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use super::wire::{
    structured_error_result, ListRecipeQueriesArgs, ListRecipeQueriesOutput, RunRecipeQueryArgs,
    RunRecipeQueryOutput,
};
use super::{list_recipe_queries, run_recipe_query, RecipeCatalog, RecipeErrorEnvelope};
use crate::tools::GraphState;

type DynFut<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

pub(crate) const LIST_RECIPE_QUERIES_TOOL: &str = "list_recipe_queries";
pub(crate) const RUN_RECIPE_QUERY_TOOL: &str = "run_recipe_query";

const LIST_DESCRIPTION: &str = "List the boot-validated Cypher recipe catalog. Omit `recipe` for compact recipe summaries; provide it to inspect that recipe's named queries and parameter schemas.";
const RUN_DESCRIPTION: &str = "Run one exact, boot-validated, read-only Cypher recipe query with strictly validated variables. Returns all rows up to the MCP payload limit or a structured error; use cypher_query for unmatched or broader questions.";

/// Register the two catalog routes as one ownership unit.
///
/// The router's normal `add_route` operation replaces an existing route with
/// the same name. Preflight both fixed names before adding either so a domain
/// or manifest collision cannot leave a partially registered catalog.
pub(crate) fn register_recipe_query_routes(
    server: &mut McpServer,
    state: GraphState,
    catalog: Arc<RecipeCatalog>,
) -> Result<usize> {
    if catalog.is_empty() {
        return Ok(0);
    }

    let router = server.tool_router_mut();
    let collisions = [LIST_RECIPE_QUERIES_TOOL, RUN_RECIPE_QUERY_TOOL]
        .into_iter()
        .filter(|name| router.map.contains_key(*name))
        .collect::<Vec<_>>();
    if !collisions.is_empty() {
        anyhow::bail!(
            "Cypher recipe routes conflict with already-registered tool(s): {}",
            collisions.join(", ")
        );
    }

    let list_catalog = catalog.clone();
    router.add_route(ToolRoute::new_dyn(
        recipe_tool::<ListRecipeQueriesArgs, ListRecipeQueriesOutput>(
            LIST_RECIPE_QUERIES_TOOL,
            LIST_DESCRIPTION,
        ),
        move |ctx: ToolCallContext<'_, McpServer>| -> DynFut<'_, Result<CallToolResponse, McpError>> {
            let catalog = list_catalog.clone();
            let arguments = ctx.arguments.clone();
            Box::pin(async move {
                let result = match deserialize_arguments::<ListRecipeQueriesArgs>(arguments) {
                    Ok(args) => list_recipe_queries(&catalog, args).into_call_tool_result(),
                    Err(error) => structured_error_result(error),
                };
                Ok(result.into())
            })
        },
    ));

    router.add_route(ToolRoute::new_dyn(
        recipe_tool::<RunRecipeQueryArgs, RunRecipeQueryOutput>(
            RUN_RECIPE_QUERY_TOOL,
            RUN_DESCRIPTION,
        ),
        move |ctx: ToolCallContext<'_, McpServer>| -> DynFut<'_, Result<CallToolResponse, McpError>> {
            let catalog = catalog.clone();
            let state = state.clone();
            let arguments = ctx.arguments.clone();
            Box::pin(async move {
                let result = match deserialize_arguments::<RunRecipeQueryArgs>(arguments) {
                    Ok(args) => run_recipe_query(&state, &catalog, args).into_call_tool_result(),
                    Err(error) => structured_error_result(error),
                };
                Ok(result.into())
            })
        },
    ));

    Ok(2)
}

fn recipe_tool<I, O>(name: &'static str, description: &'static str) -> Tool
where
    I: schemars::JsonSchema + 'static,
    O: schemars::JsonSchema + 'static,
{
    Tool::new_with_raw(name, Some(description.into()), Arc::new(Map::new()))
        .with_input_schema::<I>()
        .with_output_schema::<O>()
        .with_annotations(
            ToolAnnotations::new()
                .read_only(true)
                .destructive(false)
                .idempotent(true)
                .open_world(false),
        )
}

fn deserialize_arguments<T: DeserializeOwned>(
    arguments: Option<Map<String, Value>>,
) -> Result<T, RecipeErrorEnvelope> {
    serde_json::from_value(Value::Object(arguments.unwrap_or_default()))
        .map_err(|error| RecipeErrorEnvelope::invalid_request(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kglite::api::storage::StorageMode;
    use rmcp::model::{CallToolRequestParams, CallToolResult};
    use rmcp::ServiceExt;
    use serde_json::json;

    fn catalog() -> Arc<RecipeCatalog> {
        Arc::new(
            RecipeCatalog::from_manifest_value(Some(&json!({
                "review": {
                    "description": "Review operations.",
                    "queries": {
                        "empty": {
                            "description": "Return a deterministic empty result.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                                "additionalProperties": false
                            },
                            "cypher": "UNWIND [] AS value RETURN value ORDER BY value"
                        }
                    }
                }
            })))
            .expect("valid catalog"),
        )
    }

    fn assert_safe_contract(tool: &Tool) {
        let annotations = tool.annotations.as_ref().expect("annotations");
        assert_eq!(annotations.read_only_hint, Some(true));
        assert_eq!(annotations.destructive_hint, Some(false));
        assert_eq!(annotations.idempotent_hint, Some(true));
        assert_eq!(annotations.open_world_hint, Some(false));
        assert_eq!(tool.input_schema.get("type"), Some(&json!("object")));
        assert!(tool.output_schema.is_some(), "declared output schema");
    }

    fn structured_json(result: &CallToolResult) -> Value {
        let structured = result
            .structured_content
            .clone()
            .expect("structured content");
        let text = result.content[0].as_text().expect("text fallback");
        assert_eq!(text.text, structured.to_string());
        assert_eq!(
            serde_json::from_str::<Value>(&text.text).unwrap(),
            structured
        );
        structured
    }

    #[test]
    fn empty_catalog_registers_nothing() {
        let mut server = McpServer::new(Default::default());
        let before = server.tool_router_mut().map.len();
        assert_eq!(
            register_recipe_query_routes(
                &mut server,
                GraphState::default(),
                Arc::new(RecipeCatalog::default()),
            )
            .unwrap(),
            0
        );
        assert_eq!(server.tool_router_mut().map.len(), before);
    }

    #[test]
    fn registration_is_atomic_and_never_replaces_an_owner() {
        let mut server = McpServer::new(Default::default());
        server.register_typed_tool::<ListRecipeQueriesArgs, _>(
            LIST_RECIPE_QUERIES_TOOL,
            "Existing owner.",
            |_| "owned".to_string(),
        );

        let error = register_recipe_query_routes(&mut server, GraphState::default(), catalog())
            .expect_err("collision must fail");

        assert!(error.to_string().contains(LIST_RECIPE_QUERIES_TOOL));
        assert_eq!(
            server
                .tool_router_mut()
                .get(LIST_RECIPE_QUERIES_TOOL)
                .and_then(|tool| tool.description.as_deref()),
            Some("Existing owner.")
        );
        assert!(!server
            .tool_router_mut()
            .map
            .contains_key(RUN_RECIPE_QUERY_TOOL));
    }

    #[test]
    fn routes_publish_closed_schemas_and_all_safe_annotations() {
        let mut server = McpServer::new(Default::default());
        register_recipe_query_routes(&mut server, GraphState::default(), catalog()).unwrap();

        let router = server.tool_router_mut();
        let list = router.get(LIST_RECIPE_QUERIES_TOOL).expect("list route");
        let run = router.get(RUN_RECIPE_QUERY_TOOL).expect("run route");
        assert_safe_contract(list);
        assert_safe_contract(run);
        assert_eq!(
            list.input_schema.get("additionalProperties"),
            Some(&json!(false))
        );
        assert_eq!(
            run.input_schema.get("required"),
            Some(&json!(["recipe", "query", "variables"]))
        );
        assert_eq!(
            run.input_schema.get("additionalProperties"),
            Some(&json!(false))
        );
        let list_output = Value::Object(
            list.output_schema
                .as_ref()
                .expect("list output schema")
                .as_ref()
                .clone(),
        )
        .to_string();
        assert!(list_output.contains("recipes"));
        assert!(list_output.contains("invalid_request"));
        let run_output = Value::Object(
            run.output_schema
                .as_ref()
                .expect("run output schema")
                .as_ref()
                .clone(),
        )
        .to_string();
        assert!(run_output.contains("result"));
        for contract_field in [
            "invalid_request",
            "unknown_recipe",
            "unknown_query",
            "invalid_variables",
            "no_active_graph",
            "stale_graph",
            "query_failed",
            "result_limit_exceeded",
            "failure_message",
            "observed_count",
            "limit",
        ] {
            assert!(
                run_output.contains(contract_field),
                "run output schema is missing {contract_field:?}"
            );
        }
    }

    #[tokio::test]
    async fn real_handshake_preserves_structured_empty_and_invalid_request_results() {
        let temp = tempfile::tempdir().expect("tempdir");
        let state = GraphState::default();
        state
            .create_in_mode(&temp.path().join("empty.kgl"), StorageMode::Memory)
            .expect("create active graph");
        let mut server = McpServer::new(Default::default());
        register_recipe_query_routes(&mut server, state, catalog()).unwrap();

        let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
        let server_handle = tokio::spawn(async move { server.serve(server_transport).await });
        let client = ().serve(client_transport).await.expect("start MCP client");

        let listed = client.peer().list_tools(None).await.expect("list tools");
        let by_name = listed
            .tools
            .iter()
            .map(|tool| (tool.name.as_ref(), tool))
            .collect::<std::collections::HashMap<_, _>>();
        assert_safe_contract(by_name[LIST_RECIPE_QUERIES_TOOL]);
        assert_safe_contract(by_name[RUN_RECIPE_QUERY_TOOL]);

        let malformed = [
            CallToolRequestParams::new(RUN_RECIPE_QUERY_TOOL),
            CallToolRequestParams::new(RUN_RECIPE_QUERY_TOOL).with_arguments(
                json!({
                    "recipe": "review",
                    "query": "empty",
                    "variables": {},
                    "extra": true
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
            CallToolRequestParams::new(RUN_RECIPE_QUERY_TOOL).with_arguments(
                json!({"recipe": 3, "query": "empty", "variables": {}})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        ];
        for request in malformed {
            let invalid = client
                .call_tool(request)
                .await
                .expect("invalid request is a tool error value");
            assert_eq!(invalid.is_error, Some(true));
            assert_eq!(structured_json(&invalid)["code"], "invalid_request");
        }

        let list = client
            .call_tool(CallToolRequestParams::new(LIST_RECIPE_QUERIES_TOOL))
            .await
            .expect("compact listing succeeds");
        assert_eq!(list.is_error, Some(false));
        assert_eq!(structured_json(&list)["recipes"][0]["name"], "review");

        let unknown = client
            .call_tool(
                CallToolRequestParams::new(LIST_RECIPE_QUERIES_TOOL)
                    .with_arguments(json!({"recipe": "missing"}).as_object().unwrap().clone()),
            )
            .await
            .expect("unknown recipe is a tool error value");
        assert_eq!(unknown.is_error, Some(true));
        assert_eq!(structured_json(&unknown)["code"], "unknown_recipe");

        let empty = client
            .call_tool(
                CallToolRequestParams::new(RUN_RECIPE_QUERY_TOOL).with_arguments(
                    json!({"recipe": "review", "query": "empty", "variables": {}})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            )
            .await
            .expect("empty query succeeds");
        assert_eq!(empty.is_error, Some(false));
        let value = structured_json(&empty);
        assert_eq!(value["result"]["columns"], json!(["value"]));
        assert_eq!(value["result"]["rows"], json!([]));
        assert_eq!(value["result"]["row_count"], 0);

        client.cancel().await.expect("stop MCP client");
        server_handle.abort();
    }
}
