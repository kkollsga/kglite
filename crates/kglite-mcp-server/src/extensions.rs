//! Embedder-facing extension surface: the read-only domain graph handle,
//! the domain tool registry, and the boot-time registration of both the
//! manifest's YAML Cypher tools and injected domain routes.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use mcp_methods::server::{Manifest, McpServer};

use crate::tools::GraphState;
use crate::*;

/// Read-oriented handle to the graph slot shared by KGLite's MCP tools.
///
/// Clones remain attached to graph activation and reloads. Domain handlers can
/// inspect schema, borrow the active graph for a typed read, or execute a
/// parameterised read query without gaining access to server lifecycle state.
#[derive(Clone)]
pub struct DomainGraphState {
    inner: GraphState,
}

/// One coherent, read-only view of the active graph and its path identity.
///
/// Values are valid only for the enclosing
/// [`DomainGraphState::with_context`] callback. The graph, persistence target,
/// and source root are borrowed under one active-slot read lock, so a
/// concurrent activation cannot mix identity from one graph with another.
#[derive(Clone, Copy)]
pub struct DomainGraphContext<'a> {
    graph: &'a kglite::api::KnowledgeGraph,
    source_path: Option<&'a Path>,
    root: Option<&'a Path>,
}

impl<'a> DomainGraphContext<'a> {
    /// The active graph, borrowed read-only for the callback duration.
    pub fn graph(&self) -> &'a kglite::api::KnowledgeGraph {
        self.graph
    }

    /// Canonical persistence target used by `save_graph`, when one exists.
    pub fn source_path(&self) -> Option<&'a Path> {
        self.source_path
    }

    /// Canonical source identity: a source directory or loaded `.kgl` path.
    pub fn root(&self) -> Option<&'a Path> {
        self.root
    }
}

impl DomainGraphState {
    /// Current `(node_count, edge_count)`, or `None` when no graph is active.
    pub fn schema(&self) -> Option<(u64, u64)> {
        self.inner.schema()
    }

    /// Whether the active graph declares at least one node of this type.
    pub fn has_node_type(&self, node_type: &str) -> bool {
        self.inner.has_node_type(node_type)
    }

    /// Whether the active graph declares this property on the node type.
    pub fn has_property(&self, node_type: &str, property: &str) -> bool {
        self.inner.has_property(node_type, property)
    }

    /// Borrow the active graph for a typed, read-only operation.
    pub fn with_graph<F, T>(&self, operation: F) -> Option<T>
    where
        F: FnOnce(&kglite::api::KnowledgeGraph) -> T,
    {
        self.inner.with_kg(operation)
    }

    /// Borrow the active graph and its canonical path identity atomically.
    ///
    /// Keep the callback read-oriented and short: graph activation waits until
    /// it returns. Use this instead of separate graph/path reads whenever a
    /// domain result names the graph or source it inspected.
    pub fn with_context<F, T>(&self, operation: F) -> Option<T>
    where
        F: FnOnce(DomainGraphContext<'_>) -> T,
    {
        self.inner.with_kg_context(|graph, source_path, root| {
            operation(DomainGraphContext {
                graph,
                source_path,
                root,
            })
        })
    }

    /// Run a parameterised read query against the active graph.
    pub fn run_cypher(
        &self,
        query: &str,
        params: &serde_json::Map<String, serde_json::Value>,
    ) -> String {
        // Domain tools register through `Fn(T) -> String` (see
        // [`DomainToolRegistry::register_typed_tool`]), so the errors-as-values
        // contract is theirs: a failed read arrives as the same text a
        // successful one would have carried, for the handler to render however
        // its own tool prose calls for. KGLite's own routes take the fallible
        // path and flip `isError` instead.
        self.inner
            .run_cypher_template(query, params, &csv_http::CsvHttpState::Off)
            .unwrap_or_else(|error| error)
    }
}

/// Boot-time registry exposed to downstream domain-tool extensions.
///
/// The registry deliberately exposes registration rather than the whole
/// [`McpServer`]: downstreams can add their own tools and close over the active
/// [`GraphState`], but cannot replace KGLite's generic graph/Cypher routes.
pub struct DomainToolRegistry<'a> {
    server: &'a mut McpServer,
    graph_state: DomainGraphState,
}

impl DomainToolRegistry<'_> {
    /// The live graph slot shared by KGLite's built-in tools.
    ///
    /// Clone this cheap handle into domain-tool handlers. It follows graph
    /// activation and reloads, unlike capturing the graph active at boot.
    pub fn graph_state(&self) -> &DomainGraphState {
        &self.graph_state
    }

    /// Register a typed domain tool without allowing an existing route to be
    /// replaced. The handler follows `mcp-methods`' errors-as-values contract.
    pub fn register_typed_tool<T, F>(
        &mut self,
        name: &'static str,
        description: &'static str,
        handler: F,
    ) -> Result<()>
    where
        T: for<'de> serde::Deserialize<'de>
            + schemars::JsonSchema
            + Default
            + Send
            + Sync
            + 'static,
        F: Fn(T) -> String + Send + Sync + 'static,
    {
        self.ensure_name_available(name)?;
        self.server
            .register_typed_tool::<T, F>(name, description, handler);
        Ok(())
    }

    /// Register a fully custom rmcp route, for handlers that need capabilities
    /// beyond [`register_typed_tool`](Self::register_typed_tool).
    pub fn register_route(
        &mut self,
        route: rmcp::handler::server::router::tool::ToolRoute<McpServer>,
    ) -> Result<()> {
        self.ensure_name_available(route.attr.name.as_ref())?;
        self.server.tool_router_mut().add_route(route);
        Ok(())
    }

    pub(crate) fn ensure_name_available(&mut self, name: &str) -> Result<()> {
        if self.server.tool_router_mut().has_route(name) {
            anyhow::bail!(
                "downstream domain tool {name:?} conflicts with an already-registered KGLite or manifest tool"
            );
        }
        Ok(())
    }
}

/// Downstream callback invoked once after KGLite and manifest tools are
/// registered and before skills are finalised and the stdio server starts.
pub type DomainToolRegistrar =
    dyn for<'a> Fn(&mut DomainToolRegistry<'a>) -> Result<()> + Send + Sync + 'static;

/// Optional boot-time extension points for binaries embedding this server.
#[derive(Default)]
pub struct ServerExtensions {
    pub(crate) workspace_graph: Option<WorkspaceGraphHooks>,
    pub(crate) domain_tools: Option<Box<DomainToolRegistrar>>,
}

impl ServerExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inject an external producer for workspace build/watch paths.
    pub fn with_workspace_graph(mut self, hooks: WorkspaceGraphHooks) -> Self {
        self.workspace_graph = Some(hooks);
        self
    }

    /// Register downstream-owned domain tools against KGLite's live graph.
    pub fn with_domain_tools<F>(mut self, registrar: F) -> Self
    where
        F: for<'a> Fn(&mut DomainToolRegistry<'a>) -> Result<()> + Send + Sync + 'static,
    {
        self.domain_tools = Some(Box::new(registrar));
        self
    }
}

pub(crate) fn register_domain_tools(
    server: &mut McpServer,
    graph_state: GraphState,
    registrar: Option<Box<DomainToolRegistrar>>,
) -> Result<()> {
    if let Some(registrar) = registrar {
        let mut registry = DomainToolRegistry {
            server,
            graph_state: DomainGraphState { inner: graph_state },
        };
        registrar(&mut registry)?;
    }
    Ok(())
}

pub(crate) fn register_extension_tools(
    server: &mut McpServer,
    graph_state: &GraphState,
    manifest: Option<&Manifest>,
    csv_http: &Arc<csv_http::CsvHttpState>,
    recipe_catalog: Arc<recipe_queries::RecipeCatalog>,
    domain_tools: Option<Box<DomainToolRegistrar>>,
) -> Result<()> {
    if let Some(manifest) = manifest {
        let runner = cypher_tools::make_runner(graph_state.clone(), csv_http.clone());
        let registered = cypher_tools::register_cypher_tools(server, manifest, runner)
            .context("YAML cypher tool registration failed")?;
        if registered > 0 {
            tracing::info!(count = registered, "manifest cypher tools registered");
        }
    }
    register_domain_tools(server, graph_state.clone(), domain_tools)
        .context("downstream domain-tool registration failed")?;
    let registered =
        recipe_queries::register_recipe_query_routes(server, graph_state.clone(), recipe_catalog)
            .context("Cypher recipe route registration failed")?;
    if registered > 0 {
        tracing::info!(count = registered, "Cypher recipe routes registered");
    }
    if let Some(manifest) = manifest {
        apply_bundled_tool_overrides(server, manifest)
            .context("manifest bundled-tool override failed")?;
    }
    Ok(())
}

#[cfg(test)]
mod domain_tool_tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;

    use kglite::api::storage::StorageMode;

    use mcp_methods::server::{
        serve_prompts, BundledSkill, McpServer, ServerOptions, SkillRegistry,
    };

    use crate::tools::GraphState;

    use mcp_methods::server::{SkillSource, SkillsSource};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Default, Deserialize, JsonSchema)]
    struct DomainArgs {
        value: Option<String>,
    }

    #[test]
    fn registrar_receives_live_state_and_adds_a_tool() {
        let called = Arc::new(AtomicBool::new(false));
        let called_by_registrar = called.clone();
        let extensions = ServerExtensions::new().with_domain_tools(move |registry| {
            assert!(registry.graph_state().schema().is_none());
            called_by_registrar.store(true, Ordering::SeqCst);
            registry.register_typed_tool::<DomainArgs, _>(
                "domain_test",
                "Domain test tool.",
                |args| args.value.unwrap_or_else(|| "ok".to_string()),
            )
        });
        let mut server = McpServer::new(ServerOptions::default());

        register_domain_tools(&mut server, GraphState::new(None), extensions.domain_tools)
            .expect("register downstream tool");

        assert!(called.load(Ordering::SeqCst));
        assert!(server.tool_router_mut().has_route("domain_test"));
    }

    #[test]
    fn registrar_cannot_replace_an_existing_tool() {
        let extensions = ServerExtensions::new().with_domain_tools(|registry| {
            registry.register_typed_tool::<DomainArgs, _>(
                "ping",
                "Must not replace the framework ping.",
                |_| "replacement".to_string(),
            )
        });
        let mut server = McpServer::new(ServerOptions::default());

        let error =
            register_domain_tools(&mut server, GraphState::new(None), extensions.domain_tools)
                .expect_err("duplicate route must fail");

        assert!(error.to_string().contains("conflicts"));
        assert!(server.tool_router_mut().has_route("ping"));
    }

    #[test]
    fn registered_domain_tool_satisfies_skill_predicate() {
        const DOMAIN_SKILL: &str = r#"---
name: domain_test
description: Domain test methodology.
applies_when:
  tool_registered: domain_test
---

# Domain test

Use the registered domain tool.
"#;
        let extensions = ServerExtensions::new().with_domain_tools(|registry| {
            registry.register_typed_tool::<DomainArgs, _>(
                "domain_test",
                "Domain test tool.",
                |_| "ok".to_string(),
            )
        });
        let mut server = McpServer::new(ServerOptions::default());
        register_domain_tools(&mut server, GraphState::new(None), extensions.domain_tools)
            .expect("register downstream tool");
        let skills_source = SkillsSource::Sources(vec![SkillSource::Bundled]);
        let skills = SkillRegistry::new()
            .add_bundled(BundledSkill {
                name: "domain_test",
                body: DOMAIN_SKILL,
            })
            .layer_dirs(&skills_source, std::path::Path::new("manifest.yaml"))
            .expect("enable bundled domain skill")
            .finalise()
            .expect("resolve domain skill");

        serve_prompts(&skills, &mut server);

        let prompts = server.prompt_router_mut().list_all();
        assert!(prompts.iter().any(|prompt| prompt.name == "domain_test"));
    }

    #[test]
    fn graph_context_tracks_activation_as_one_coherent_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first_path = dir.path().join("first.kgl");
        let second_path = dir.path().join("second.kgl");
        let inner = GraphState::new(None);
        inner
            .create_in_mode(&first_path, StorageMode::Memory)
            .expect("activate first graph");
        let state = DomainGraphState {
            inner: inner.clone(),
        };

        let first = state
            .with_context(|context| {
                (
                    Arc::as_ptr(context.graph().dir()),
                    context.source_path().map(Path::to_path_buf),
                    context.root().map(Path::to_path_buf),
                )
            })
            .expect("first context");
        assert_eq!(first.1.as_deref(), Some(first_path.as_path()));
        assert_eq!(first.2.as_deref(), Some(first_path.as_path()));

        inner
            .create_in_mode(&second_path, StorageMode::Memory)
            .expect("activate second graph");
        let second = state
            .with_context(|context| {
                (
                    Arc::as_ptr(context.graph().dir()),
                    context.source_path().map(Path::to_path_buf),
                    context.root().map(Path::to_path_buf),
                )
            })
            .expect("second context");
        assert_ne!(first.0, second.0);
        assert_eq!(second.1.as_deref(), Some(second_path.as_path()));
        assert_eq!(second.2.as_deref(), Some(second_path.as_path()));
    }
}
