//! Manifest `tools:` overrides applied to already-registered bundled
//! routes — hide, rename, and description replacement.

use anyhow::Result;
use mcp_methods::server::manifest::ToolSpec;
use mcp_methods::server::{Manifest, McpServer};

use crate::*;

/// Apply manifest overrides only after every route source has registered.
///
/// A bundled override may target a framework, KGLite, manifest, or downstream
/// route, so applying it any earlier lets later registration silently undo a
/// hidden flag or bypass rename/description validation.
pub(crate) fn apply_bundled_tool_overrides(
    server: &mut McpServer,
    manifest: &Manifest,
) -> Result<()> {
    let overrides: Vec<_> = manifest
        .tools
        .iter()
        .filter_map(|tool| match tool {
            ToolSpec::Bundled(override_) => Some(override_),
            ToolSpec::Cypher(_) | ToolSpec::Python(_) => None,
        })
        .collect();
    if overrides.is_empty() {
        return Ok(());
    }

    for override_ in &overrides {
        let owns_fixed_recipe_route = [
            recipe_queries::LIST_RECIPE_QUERIES_TOOL,
            recipe_queries::RUN_RECIPE_QUERY_TOOL,
        ]
        .contains(&override_.name.as_str());
        if owns_fixed_recipe_route && (override_.hidden || override_.rename.is_some()) {
            anyhow::bail!(
                "fixed Cypher recipe route {:?} cannot be hidden or renamed",
                override_.name
            );
        }
        if let Some(rename) = override_.rename.as_deref() {
            if [
                recipe_queries::LIST_RECIPE_QUERIES_TOOL,
                recipe_queries::RUN_RECIPE_QUERY_TOOL,
            ]
            .contains(&rename)
                && rename != override_.name
            {
                anyhow::bail!(
                    "manifest bundled-tool route {:?} cannot be renamed onto fixed Cypher recipe route {rename:?}",
                    override_.name
                );
            }
        }
    }

    let router = server.tool_router_mut();
    for override_ in &overrides {
        if !router.map.contains_key(override_.name.as_str()) {
            anyhow::bail!(
                "manifest bundled-tool override targets unknown route {:?}",
                override_.name
            );
        }
    }

    let final_names = overrides
        .iter()
        .map(|override_| {
            (
                override_.name.as_str(),
                override_.rename.as_deref().unwrap_or(&override_.name),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut owners = std::collections::HashMap::<&str, &str>::new();
    for name in router.map.keys().map(|name| name.as_ref()) {
        let final_name = final_names.get(name).copied().unwrap_or(name);
        if let Some(existing) = owners.insert(final_name, name) {
            anyhow::bail!(
                "manifest bundled-tool rename {final_name:?} conflicts between routes {existing:?} and {name:?}"
            );
        }
    }

    let mut routes = Vec::with_capacity(overrides.len());
    for override_ in overrides {
        let was_disabled = router.is_disabled(&override_.name);
        let route = router
            .map
            .remove(override_.name.as_str())
            .expect("override targets validated above");
        routes.push((override_, route, was_disabled));
    }

    for (override_, mut route, was_disabled) in routes {
        let final_name = override_.rename.as_deref().unwrap_or(&override_.name);
        route.attr.name = final_name.to_owned().into();
        if let Some(description) = &override_.description {
            route.attr.description = Some(description.clone().into());
        }
        router.add_route(route);
        if override_.hidden || was_disabled {
            router.disable_route(final_name.to_owned());
        }
    }

    Ok(())
}

#[cfg(test)]
mod bundled_override_tests {
    use super::*;

    use std::sync::Arc;

    use mcp_methods::server::{Manifest, McpServer, ServerOptions};

    use rmcp::ServiceExt;

    use crate::tools::GraphState;

    use rmcp::model::CallToolRequestParams;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Default, Deserialize, JsonSchema)]
    struct CollisionArgs {}

    fn load_manifest(yaml: &str) -> (tempfile::TempDir, Manifest) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mcp.yaml");
        std::fs::write(&path, yaml).expect("write manifest");
        let manifest = mcp_methods::server::load_manifest(&path).expect("load manifest");
        (dir, manifest)
    }

    #[tokio::test]
    async fn hidden_route_is_unlisted_and_rejects_direct_calls() {
        let (_dir, manifest) = load_manifest("tools:\n  - bundled: ping\n    hidden: true\n");
        let mut server = McpServer::new(ServerOptions::default());

        apply_bundled_tool_overrides(&mut server, &manifest).expect("apply override");

        assert!(!server.tool_router_mut().has_route("ping"));
        assert!(server.tool_router_mut().is_disabled("ping"));
        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server_handle = tokio::spawn(async move { server.serve(server_transport).await });
        let client = ().serve(client_transport).await.expect("start MCP client");

        let listed = client.peer().list_tools(None).await.expect("list tools");
        assert!(listed.tools.iter().all(|tool| tool.name != "ping"));
        let error = client
            .call_tool(CallToolRequestParams::new("ping"))
            .await
            .expect_err("hidden route must reject direct calls");
        assert!(error.to_string().contains("tool not found"));

        client.cancel().await.expect("stop MCP client");
        server_handle.abort();
    }

    #[test]
    fn rename_and_description_replace_the_agent_facing_route() {
        let (_dir, manifest) = load_manifest(
            "tools:\n  - bundled: ping\n    rename: domain_ping\n    description: Domain health check.\n",
        );
        let mut server = McpServer::new(ServerOptions::default());

        apply_bundled_tool_overrides(&mut server, &manifest).expect("apply override");

        assert!(!server.tool_router_mut().has_route("ping"));
        let renamed = server
            .tool_router_mut()
            .get("domain_ping")
            .expect("renamed route");
        assert_eq!(renamed.description.as_deref(), Some("Domain health check."));
    }

    #[test]
    fn unknown_and_colliding_routes_fail_at_boot() {
        let (_dir, unknown) =
            load_manifest("tools:\n  - bundled: not_a_bundled_route\n    hidden: true\n");
        let mut server = McpServer::new(ServerOptions::default());
        let error = apply_bundled_tool_overrides(&mut server, &unknown)
            .expect_err("unknown route must fail");
        assert!(error.to_string().contains("unknown route"));

        let (_dir, collision) =
            load_manifest("tools:\n  - bundled: ping\n    rename: list_source\n");
        let mut server = McpServer::new(ServerOptions::default());
        let error = apply_bundled_tool_overrides(&mut server, &collision)
            .expect_err("rename collision must fail");
        assert!(error.to_string().contains("conflicts"));
    }

    /// The local-workspace boot path hides `repo_management` before manifest
    /// overrides run. It must hide it by *disabling* the route, because the
    /// unknown-route bail above reads `router.map`: a removed route makes any
    /// manifest naming it a hard boot error, while a disabled one resolves.
    /// Both halves are asserted so the constraint cannot silently invert.
    #[test]
    fn hiding_a_route_before_overrides_must_disable_it_not_remove_it() {
        let manifest_yaml = "tools:\n  - bundled: repo_management\n    hidden: true\n";

        fn workspace_server() -> McpServer {
            let mut server = McpServer::new(ServerOptions::default());
            server.register_typed_tool::<CollisionArgs, _>(
                "repo_management",
                "Stub repo-management route.",
                |_| "stub".to_string(),
            );
            server
        }

        // Pre-fix boot shape: the route is gone from `map`, so the manifest
        // override targets a name the router no longer knows.
        let (_dir, manifest) = load_manifest(manifest_yaml);
        let mut removed = workspace_server();
        removed.tool_router_mut().remove_route("repo_management");
        let error = apply_bundled_tool_overrides(&mut removed, &manifest)
            .expect_err("a removed route must still be rejected as unknown");
        assert!(error.to_string().contains("unknown route"));

        // Fixed boot shape: same user-visible effect, override resolves.
        let (_dir, manifest) = load_manifest(manifest_yaml);
        let mut disabled = workspace_server();
        disabled.tool_router_mut().disable_route("repo_management");
        assert!(disabled.tool_router_mut().is_disabled("repo_management"));

        apply_bundled_tool_overrides(&mut disabled, &manifest)
            .expect("a manifest may hide an already-disabled route");

        assert!(
            disabled.tool_router_mut().is_disabled("repo_management"),
            "the hide override must leave the route disabled, not re-enable it"
        );
        assert!(!disabled.tool_router_mut().has_route("repo_management"));
    }

    fn recipe_catalog() -> Arc<recipe_queries::RecipeCatalog> {
        Arc::new(
            recipe_queries::RecipeCatalog::from_manifest_value(Some(&json!({
                "review": {
                    "description": "Review operations.",
                    "queries": {
                        "lookup": {
                            "description": "Look up one value.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                                "additionalProperties": false
                            },
                            "cypher": "RETURN 1 AS value"
                        }
                    }
                }
            })))
            .expect("valid recipe catalog"),
        )
    }

    #[test]
    fn recipe_routes_allow_description_only_overrides() {
        let (_dir, manifest) = load_manifest(
            "tools:\n  - bundled: run_recipe_query\n    description: Domain recipe runner.\n",
        );
        let mut server = McpServer::new(ServerOptions::default());
        recipe_queries::register_recipe_query_routes(
            &mut server,
            GraphState::default(),
            recipe_catalog(),
        )
        .expect("register recipe routes");

        apply_bundled_tool_overrides(&mut server, &manifest).expect("description override");

        assert_eq!(
            server
                .tool_router_mut()
                .get(recipe_queries::RUN_RECIPE_QUERY_TOOL)
                .and_then(|tool| tool.description.as_deref()),
            Some("Domain recipe runner.")
        );
    }

    #[test]
    fn fixed_recipe_routes_reject_hide_rename_and_rename_onto() {
        for yaml in [
            "tools:\n  - bundled: list_recipe_queries\n    hidden: true\n",
            "tools:\n  - bundled: run_recipe_query\n    rename: domain_recipe_runner\n",
        ] {
            let (_dir, manifest) = load_manifest(yaml);
            let mut server = McpServer::new(ServerOptions::default());
            recipe_queries::register_recipe_query_routes(
                &mut server,
                GraphState::default(),
                recipe_catalog(),
            )
            .expect("register recipe routes");
            let error = apply_bundled_tool_overrides(&mut server, &manifest)
                .expect_err("fixed route ownership must be stable");
            assert!(error.to_string().contains("cannot be hidden or renamed"));
        }

        let (_dir, manifest) =
            load_manifest("tools:\n  - bundled: ping\n    rename: run_recipe_query\n");
        let mut server = McpServer::new(ServerOptions::default());
        let error = apply_bundled_tool_overrides(&mut server, &manifest)
            .expect_err("another route cannot claim a fixed recipe name");
        assert!(error.to_string().contains("cannot be renamed onto"));
    }

    #[test]
    fn recipe_registration_detects_legacy_manifest_owner_after_it_registers() {
        let (_dir, manifest) = load_manifest(
            "tools:\n  - name: run_recipe_query\n    description: Legacy owner.\n    cypher: RETURN 1 AS value\n",
        );
        let mut server = McpServer::new(ServerOptions::default());

        let error = register_extension_tools(
            &mut server,
            &GraphState::default(),
            Some(&manifest),
            &std::sync::Arc::default(),
            recipe_catalog(),
            None,
        )
        .expect_err("legacy owner must block recipe registration");

        assert!(format!("{error:#}").contains(recipe_queries::RUN_RECIPE_QUERY_TOOL));
        assert_eq!(
            server
                .tool_router_mut()
                .get(recipe_queries::RUN_RECIPE_QUERY_TOOL)
                .and_then(|tool| tool.description.as_deref()),
            Some("Legacy owner.")
        );
        assert!(
            !server
                .tool_router_mut()
                .map
                .contains_key(recipe_queries::LIST_RECIPE_QUERIES_TOOL),
            "atomic recipe registration must not add the other fixed route"
        );
    }

    #[test]
    fn recipe_registration_detects_domain_owner_after_it_registers() {
        let mut server = McpServer::new(ServerOptions::default());
        let domain_tools: Box<DomainToolRegistrar> = Box::new(|registry| {
            registry.register_typed_tool::<CollisionArgs, _>(
                recipe_queries::LIST_RECIPE_QUERIES_TOOL,
                "Domain owner.",
                |_| "owned".to_string(),
            )
        });

        let error = register_extension_tools(
            &mut server,
            &GraphState::default(),
            None,
            &std::sync::Arc::default(),
            recipe_catalog(),
            Some(domain_tools),
        )
        .expect_err("domain owner must block recipe registration");

        assert!(format!("{error:#}").contains(recipe_queries::LIST_RECIPE_QUERIES_TOOL));
        assert_eq!(
            server
                .tool_router_mut()
                .get(recipe_queries::LIST_RECIPE_QUERIES_TOOL)
                .and_then(|tool| tool.description.as_deref()),
            Some("Domain owner.")
        );
        assert!(
            !server
                .tool_router_mut()
                .map
                .contains_key(recipe_queries::RUN_RECIPE_QUERY_TOOL),
            "atomic recipe registration must not add the other fixed route"
        );
    }
}
