//! Downstream contract tests for the mcp-methods response-budget boundary.
//!
//! These deliberately exercise public APIs and real protocol dispatch. They
//! prevent a dependency update from compiling while silently dropping the
//! discovery, non-replay, or structured-selection behavior KGLite builds on.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use mcp_methods::response_budget::{Expansion, Mode, ResponseOptions, ResponseStore};
use mcp_methods::server::{McpServer, ServerOptions};
use rmcp::handler::server::router::tool::ToolRoute;
use rmcp::model::{CallToolResult, Tool};
use rmcp::ServiceExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Args {
    value: String,
}

fn preview(result: &CallToolResult) -> Value {
    let text = result.content[0].as_text().expect("text preview");
    serde_json::from_str::<Value>(&text.text).expect("JSON preview")["response_budget"].clone()
}

async fn call(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    name: &str,
    arguments: Value,
) -> CallToolResult {
    let params = match arguments {
        Value::Object(map) => {
            rmcp::model::CallToolRequestParams::new(name.to_string()).with_arguments(map)
        }
        _ => rmcp::model::CallToolRequestParams::new(name.to_string()),
    };
    client.call_tool(params).await.expect("tool call")
}

async fn boot(server: McpServer) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let (server_transport, client_transport) = tokio::io::duplex(1024 * 1024);
    tokio::spawn(async move {
        let service = server.serve(server_transport).await.expect("serve");
        let _ = service.waiting().await;
    });
    ().serve(client_transport).await.expect("start MCP client")
}

#[tokio::test]
async fn dispatch_discovery_overrides_and_expansion_are_live() {
    let source = tempfile::tempdir().expect("source root");
    std::fs::write(
        source.path().join("large.txt"),
        "builtin evidence\n".repeat(4_000),
    )
    .expect("large builtin fixture");
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let large = json!({
        "columns": ["id", "body"],
        "rows": (0..180).map(|i| json!([i, "x".repeat(400)])).collect::<Vec<_>>(),
        "diagnostics": {"warnings": ["database-wide coverage unknown"]}
    });
    let large_text = large.to_string();
    let typed_output = large_text.clone();

    let mut server = McpServer::new(
        ServerOptions::default()
            .with_static_source_roots(vec![source.path().to_string_lossy().into_owned()]),
    );
    server.register_typed_tool("small", "Small result", |args: Args| args.value);
    server.register_typed_tool("mutate", "Count once", move |_: Args| {
        counted.fetch_add(1, Ordering::SeqCst);
        typed_output.clone()
    });
    server.register_typed_tool_fallible("fail", "Large error", |_: Args| {
        Err("failed with evidence\n".repeat(2_000))
    });
    let custom = Tool::new(
        "structured",
        "Structured result",
        Arc::new(json!({"type":"object"}).as_object().unwrap().clone()),
    );
    server
        .tool_router_mut()
        .add_route(ToolRoute::new_dyn(custom, move |_| {
            let value = large.clone();
            Box::pin(async move { Ok(CallToolResult::structured(value).into()) })
        }));
    let custom_error = Tool::new(
        "custom_error",
        "Custom error result",
        Arc::new(json!({"type":"object"}).as_object().unwrap().clone()),
    );
    server
        .tool_router_mut()
        .add_route(ToolRoute::new_dyn(custom_error, |_| {
            Box::pin(async {
                Ok(CallToolResult::error(vec![rmcp::model::ContentBlock::text(
                    "custom failure evidence\n".repeat(2_000),
                )])
                .into())
            })
        }));

    let client = boot(server).await;
    let listed = client.list_tools(None).await.expect("list tools");
    let mutate = listed
        .tools
        .iter()
        .find(|tool| tool.name == "mutate")
        .unwrap();
    assert_eq!(
        mutate.input_schema["properties"]["_response"]["properties"]["mode"]["default"],
        "bounded"
    );
    assert!(listed
        .tools
        .iter()
        .any(|tool| tool.name == "expand_response"));

    let small = call(&client, "small", json!({"value":"ok"})).await;
    assert_eq!(small.content[0].as_text().unwrap().text, "ok");
    assert!(small.structured_content.is_none());

    let invalid = client
        .call_tool(
            rmcp::model::CallToolRequestParams::new("mutate").with_arguments(
                json!({"value":"x", "_response":{"max_bytes":1}})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        )
        .await;
    assert!(
        invalid.is_err(),
        "invalid control must fail before dispatch"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let bounded = call(&client, "mutate", json!({"value":"x"})).await;
    let bounded_json = serde_json::to_vec(&bounded).unwrap();
    assert!(bounded_json.len() <= 16_384, "{} bytes", bounded_json.len());
    let guide = preview(&bounded);
    let full = &guide["next"]["full_result"];
    let recovered = call(
        &client,
        full["name"].as_str().unwrap(),
        full["arguments"].clone(),
    )
    .await;
    assert_eq!(recovered.content[0].as_text().unwrap().text, large_text);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "expansion reran mutation");

    let larger = call(
        &client,
        "mutate",
        json!({"value":"x", "_response":{"max_bytes":32768}}),
    )
    .await;
    assert_eq!(preview(&larger)["max_bytes"], 32_768);
    let complete = call(
        &client,
        "mutate",
        json!({"value":"x", "_response":{"mode":"full"}}),
    )
    .await;
    assert_eq!(complete.content[0].as_text().unwrap().text, large_text);

    let failed = call(&client, "fail", json!({"value":"x"})).await;
    assert_eq!(failed.is_error, Some(true));
    let failed_full = preview(&failed)["next"]["full_result"].clone();
    let failed_recovered = call(
        &client,
        failed_full["name"].as_str().unwrap(),
        failed_full["arguments"].clone(),
    )
    .await;
    assert_eq!(failed_recovered.is_error, Some(true));
    assert!(failed_recovered.content[0]
        .as_text()
        .unwrap()
        .text
        .starts_with("failed with evidence"));

    let structured = call(&client, "structured", json!({})).await;
    let structured_full = preview(&structured)["next"]["full_result"].clone();
    let structured_recovered = call(
        &client,
        structured_full["name"].as_str().unwrap(),
        structured_full["arguments"].clone(),
    )
    .await;
    assert_eq!(
        structured_recovered.structured_content.unwrap()["rows"][179][0],
        179
    );

    for (name, arguments, expected_error, marker) in [
        (
            "read_source",
            json!({"file_path":"large.txt"}),
            false,
            "builtin evidence",
        ),
        ("custom_error", json!({}), true, "custom failure evidence"),
    ] {
        let bounded = call(&client, name, arguments).await;
        assert!(serde_json::to_vec(&bounded).unwrap().len() <= 16_384);
        assert_eq!(bounded.is_error == Some(true), expected_error);
        let action = preview(&bounded)["next"]["full_result"].clone();
        let restored = call(
            &client,
            action["name"].as_str().unwrap(),
            action["arguments"].clone(),
        )
        .await;
        assert_eq!(restored.is_error == Some(true), expected_error);
        assert!(restored.content[0].as_text().unwrap().text.contains(marker));
    }

    client.cancel().await.expect("stop client");
}

#[tokio::test]
async fn discovery_names_controls_and_expansion_around_domain_collisions() {
    #[derive(Default, Deserialize, JsonSchema)]
    struct CollisionArgs {
        #[serde(rename = "_response")]
        domain_value: String,
    }

    let mut server = McpServer::new(ServerOptions::default());
    server.register_typed_tool("expand_response", "Domain tool", |args: CollisionArgs| {
        args.domain_value
    });
    server.register_typed_tool("large", "Large result", |_: Args| "z".repeat(40_000));
    let client = boot(server).await;
    let listed = client.list_tools(None).await.expect("list tools");
    let domain = listed
        .tools
        .iter()
        .find(|tool| tool.name == "expand_response")
        .unwrap();
    assert!(domain.input_schema["properties"]
        .get("_response_")
        .is_some());
    assert!(listed
        .tools
        .iter()
        .any(|tool| tool.name == "expand_response_"));

    let domain_result = call(
        &client,
        "expand_response",
        json!({"_response":"domain value", "_response_":{"mode":"full"}}),
    )
    .await;
    assert_eq!(
        domain_result.content[0].as_text().unwrap().text,
        "domain value"
    );

    let result = call(&client, "large", json!({"value":"x"})).await;
    let advertised = preview(&result)["next"]["full_result"].clone();
    assert_eq!(advertised["name"], "expand_response_");
    let expanded = call(
        &client,
        advertised["name"].as_str().unwrap(),
        advertised["arguments"].clone(),
    )
    .await;
    assert_eq!(expanded.content[0].as_text().unwrap().text.len(), 40_000);
    client.cancel().await.expect("stop client");
}

#[test]
fn cargo_lock_resolves_the_response_contract_release() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
        .expect("workspace Cargo.lock");
    let package = lock
        .split("[[package]]")
        .find(|entry| entry.contains("name = \"mcp-methods\""))
        .expect("mcp-methods lock entry");
    assert!(package.contains("version = \"0.4.9\""), "{package}");
}

/// Model the Phase 4/5 adapter without persisting framework-private records:
/// serialize KGLite's canonical envelope, then rehydrate a fresh public store.
#[test]
fn public_store_rehydrates_structured_cli_evidence_and_handles_direct_results() {
    let canonical = json!({
        "schema_version": 1,
        "kind": "cypher_result",
        "columns": ["id", "evidence"],
        "rows": (0..100).map(|i| json!([i, {"body":"界".repeat(300)}])).collect::<Vec<_>>(),
        "diagnostics": {"warnings": ["query LIMIT makes graph population unknown"]},
        "coverage": {"executed_rows":100, "database_population":"unknown"},
        "identity": {"workspace":"digest"},
        "operation": {"mutation":true}
    });
    let persisted = serde_json::to_vec(&canonical).expect("persist canonical envelope");
    let reopened: Value = serde_json::from_slice(&persisted).expect("new process reads envelope");
    let original = json!({
        "content":[{"type":"text","text":"agent result"}],
        "structuredContent": reopened,
        "isError":false
    });

    let mut first = ResponseStore::default();
    let first_preview = first.present(
        "rehydrated-owner",
        "kglite query",
        json!({"scope":"stored; query is not executable"}),
        original.clone(),
        &ResponseOptions::default(),
        "expand_response",
    );
    let first_guide: Value =
        serde_json::from_str(first_preview["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(first_guide["response_budget"]["preview"]["path"], "");
    assert!(first_guide["response_budget"]["preview"]
        .to_string()
        .contains("/rows"));

    // A later process creates a fresh store from only the canonical result.
    let mut later = ResponseStore::default();
    let later_preview = later.present(
        "rehydrated-owner",
        "kglite query",
        json!({"scope":"stored; graph may be gone"}),
        original,
        &ResponseOptions::default(),
        "expand_response",
    );
    let later_guide: Value =
        serde_json::from_str(later_preview["content"][0]["text"].as_str().unwrap()).unwrap();
    let temporary_id = later_guide["response_budget"]["result_id"]
        .as_str()
        .unwrap();
    let selected = later
        .expand(
            &"rehydrated-owner",
            &Expansion {
                result_id: temporary_id.to_string(),
                path: "/rows/80/1/body".to_string(),
                offset: 0,
                response: ResponseOptions {
                    mode: Mode::Full,
                    max_bytes: None,
                },
            },
            "expand_response",
        )
        .expect("select nested value after process lifetime");
    assert_eq!(
        selected["content"][0]["text"],
        json!("界".repeat(300)).to_string()
    );

    let small = json!({
        "content":[{"type":"text","text":"small"}],
        "structuredContent":{"rows":[[1]]},
        "isError":false
    });
    let mut direct_store = ResponseStore::default();
    let direct = direct_store.present(
        "owner",
        "query",
        Value::Null,
        small.clone(),
        &ResponseOptions::default(),
        "expand_response",
    );
    assert_eq!(direct, small, "fitting result has no temporary handle");

    let mut full_store = ResponseStore::default();
    let full = full_store.present(
        "owner",
        "query",
        Value::Null,
        small.clone(),
        &ResponseOptions {
            mode: Mode::Full,
            max_bytes: None,
        },
        "expand_response",
    );
    assert_eq!(full, small, "full mode has no temporary handle");
    assert!(ResponseOptions {
        mode: Mode::Bounded,
        max_bytes: Some(4_095)
    }
    .validate()
    .is_err());
    assert!(ResponseOptions {
        mode: Mode::Bounded,
        max_bytes: Some(4_096)
    }
    .validate()
    .is_ok());
}
