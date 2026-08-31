//! `extensions.tools_allow:` — the closed-by-default tool surface.
//!
//! A deployment that serves one domain graph usually wants a handful of tools,
//! but the server's surface is the *union* of everything that happened to
//! register: framework builtins, source tools bound by the mode, credential-gated
//! routes, KGLite's graph tools, manifest Cypher tools and downstream domain
//! routes. Naming the ones to *drop* is a losing game: the list grows from the
//! outside.
//!
//! One such source has since been closed upstream — before mcp-methods 0.4.5 an
//! ambient `GITHUB_TOKEN` in the operator's environment registered the GitHub
//! tools without the manifest ever mentioning them; they now need
//! `builtins.github: true`. That is one route fixed at the source, not the
//! general problem: any dependency can still add a route this deployment never
//! asked for.
//!
//! `extensions.tools_allow` inverts it. The manifest names the final tool surface
//! and everything else is disabled — so a new route arriving from a dependency,
//! an environment variable, or a mode change cannot widen a deployment's agent
//! surface without an edit to that list.

use anyhow::Result;
use mcp_methods::server::McpServer;

use crate::*;

/// Disable every registered route the allowlist does not name.
///
/// Applied once, after **every** route source has registered *and* after
/// [`apply_bundled_tool_overrides`] has run, because the names in the list are
/// the ones an agent sees: a `rename:` override makes `ping` reachable only as
/// `domain_ping`, and matching before the rename would keep the wrong route.
///
/// Three properties are deliberate:
///
/// 1. **Naming an absent route is a no-op, never an error.** Half the routes
///    worth allowing are conditional — GitHub tools appear only when the
///    manifest opts in and a token is reachable,
///    `explore`/`read_code_source` only on a code graph, `save_graph`/`load_graph`
///    only on a write-enabled server. A list that errored on the absent ones
///    would force operators to maintain a different manifest per environment,
///    which is exactly the fragility this key exists to remove.
/// 2. **It only removes.** A route another mechanism disabled (the
///    non-code-graph code-tool gate,
///    a manifest `hidden: true`) stays disabled even when the allowlist names
///    it. `enable_route` is never called here: an allowlist expresses the
///    ceiling of the surface, not the floor.
/// 3. **`disable_route`, never `remove_route`.** A removed route is gone from
///    `router.map`, and a manifest override naming it then hard-errors at boot
///    ("unknown route"); disabling produces the same user-visible effect —
///    unlisted, rejected on call — while leaving overrides resolvable.
///
/// The one refusal: a manifest that configures `extensions.cypher_recipes` and
/// then omits the two fixed recipe routes is contradictory, and is rejected for
/// the same reason those routes cannot be hidden or renamed — the catalog's
/// discovery/run pair is one owned unit.
pub(crate) fn apply_tool_allowlist(
    server: &mut McpServer,
    allow: &[String],
    recipes_configured: bool,
) -> Result<()> {
    let allowed = allow
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let router = server.tool_router_mut();

    if recipes_configured {
        for fixed in [
            recipe_queries::LIST_RECIPE_QUERIES_TOOL,
            recipe_queries::RUN_RECIPE_QUERY_TOOL,
        ] {
            if router.map.contains_key(fixed) && !allowed.contains(fixed) {
                anyhow::bail!(
                    "extensions.tools_allow omits fixed Cypher recipe route {fixed:?}: a \
                     configured recipe catalog cannot be hidden — add it to the list, or drop \
                     extensions.cypher_recipes"
                );
            }
        }
    }

    let disable = router
        .map
        .keys()
        .map(|name| name.as_ref())
        .filter(|name| !allowed.contains(name))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let absent = allow
        .iter()
        .filter(|name| !router.map.contains_key(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    for name in &disable {
        router.disable_route(name.clone());
    }

    tracing::info!(
        allowed = allow.len(),
        disabled = disable.len(),
        "extensions.tools_allow applied"
    );
    if !absent.is_empty() {
        // Expected for conditional routes; logged so a typo is still findable.
        tracing::debug!(
            names = absent.join(", "),
            "extensions.tools_allow names routes that are not registered in this boot"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tools_allow_tests {
    use super::*;

    use std::sync::Arc;

    use mcp_methods::server::{McpServer, ServerOptions};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    use crate::tools::GraphState;

    #[derive(Default, Deserialize, JsonSchema)]
    struct StubArgs {}

    fn server_with(routes: &[&'static str]) -> McpServer {
        let mut server = McpServer::new(ServerOptions::default());
        for name in routes {
            server.register_typed_tool::<StubArgs, _>(name, "Stub route.", |_| "stub".to_string());
        }
        server
    }

    fn allow(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
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

    fn server_with_recipes() -> McpServer {
        let mut server = McpServer::new(ServerOptions::default());
        recipe_queries::register_recipe_query_routes(
            &mut server,
            GraphState::default(),
            recipe_catalog(),
        )
        .expect("register recipe routes");
        server
    }

    /// The core contract: named routes survive, every other registered route is
    /// disabled — and disabled means *disabled*, not removed, so a manifest
    /// override naming a dropped tool still resolves.
    #[test]
    fn every_unnamed_route_is_disabled_and_the_named_ones_survive() {
        let mut server = server_with(&["domain_tool", "unwanted_tool"]);
        // Framework routes the deployment never asked for are the point of the
        // feature — assert one is live before the allowlist runs, so a passing
        // test cannot be an artefact of it never having been registered.
        assert!(server.tool_router_mut().has_route("grep"));

        apply_tool_allowlist(
            &mut server,
            &allow(&["ping", "domain_tool"]),
            /* recipes_configured */ false,
        )
        .expect("allowlist applies");

        let router = server.tool_router_mut();
        assert!(router.has_route("ping"));
        assert!(router.has_route("domain_tool"));
        for dropped in ["grep", "read_source", "list_source", "unwanted_tool"] {
            assert!(
                !router.has_route(dropped),
                "{dropped} must not survive a closed allowlist"
            );
            assert!(
                router.is_disabled(dropped),
                "{dropped} must be disabled, not removed — overrides naming it must still resolve"
            );
        }
    }

    /// Conditional routes (opt-in + token-gated GitHub tools, mode-gated code
    /// tools, writable-only lifecycle tools) legitimately do not exist in
    /// every boot.
    #[test]
    fn naming_a_route_that_never_registered_is_a_no_op() {
        let mut server = server_with(&[]);

        apply_tool_allowlist(
            &mut server,
            &allow(&["ping", "github_api", "load_graph", "not_a_tool_at_all"]),
            false,
        )
        .expect("an absent name must not fail boot");

        assert!(server.tool_router_mut().has_route("ping"));
        assert!(!server.tool_router_mut().map.contains_key("github_api"));
    }

    /// A configured recipe catalog owns its two fixed routes: the allowlist may
    /// not silently drop half of the catalog contract.
    #[test]
    fn omitting_a_configured_recipe_route_refuses_at_boot() {
        for omitted in [
            allow(&["ping"]),
            allow(&["ping", recipe_queries::LIST_RECIPE_QUERIES_TOOL]),
            allow(&["ping", recipe_queries::RUN_RECIPE_QUERY_TOOL]),
        ] {
            let mut server = server_with_recipes();
            let error = apply_tool_allowlist(&mut server, &omitted, true)
                .expect_err("a configured catalog cannot be dropped by the allowlist");
            assert!(
                error.to_string().contains("fixed Cypher recipe route"),
                "{error}"
            );
        }

        let mut server = server_with_recipes();
        apply_tool_allowlist(
            &mut server,
            &allow(&[
                "ping",
                recipe_queries::LIST_RECIPE_QUERIES_TOOL,
                recipe_queries::RUN_RECIPE_QUERY_TOOL,
            ]),
            true,
        )
        .expect("naming both fixed routes is accepted");
        assert!(server
            .tool_router_mut()
            .has_route(recipe_queries::RUN_RECIPE_QUERY_TOOL));

        // No catalog configured: the names are simply absent, and omitting them
        // is the ordinary no-op — the refusal must not fire on every manifest.
        let mut server = server_with(&[]);
        apply_tool_allowlist(&mut server, &allow(&["ping"]), false)
            .expect("without a recipe catalog there is nothing to protect");
    }

    /// Overrides run first, so the list is written against the names an agent
    /// sees. Both halves are asserted: the new name keeps the route, the old
    /// name no longer refers to anything.
    #[test]
    fn matching_uses_post_rename_names() {
        let manifest_yaml = "tools:\n  - bundled: ping\n    rename: domain_ping\n";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mcp.yaml");
        std::fs::write(&path, manifest_yaml).expect("write manifest");
        let manifest = mcp_methods::server::load_manifest(&path).expect("load manifest");

        let mut renamed = server_with(&[]);
        apply_bundled_tool_overrides(&mut renamed, &manifest).expect("apply rename");
        apply_tool_allowlist(&mut renamed, &allow(&["domain_ping"]), false).expect("allowlist");
        assert!(renamed.tool_router_mut().has_route("domain_ping"));

        let mut old_name = server_with(&[]);
        apply_bundled_tool_overrides(&mut old_name, &manifest).expect("apply rename");
        apply_tool_allowlist(&mut old_name, &allow(&["ping"]), false).expect("allowlist");
        assert!(
            !old_name.tool_router_mut().has_route("domain_ping"),
            "the pre-rename name must not keep the renamed route alive"
        );
        assert!(
            !old_name.tool_router_mut().map.contains_key("ping"),
            "the pre-rename name is simply absent — an allowlist naming it is a no-op"
        );
    }

    /// The allowlist is a ceiling, not a floor: it never re-enables a route
    /// another gate disabled (the code-tool gate on a non-code graph, a
    /// manifest `hidden: true`).
    #[test]
    fn an_allowlisted_route_that_is_already_disabled_stays_disabled() {
        let mut server = server_with(&["explore"]);
        server.tool_router_mut().disable_route("explore");

        apply_tool_allowlist(&mut server, &allow(&["ping", "explore"]), false)
            .expect("allowlist applies");

        assert!(server.tool_router_mut().has_route("ping"));
        assert!(
            server.tool_router_mut().is_disabled("explore"),
            "an allowlist must never re-enable a route another gate disabled"
        );
    }
}
