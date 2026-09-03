//! Skill and discovery-steer wiring: the lazy-discovery instruction fold,
//! the conditionally bundled recipe-query skill, and the graph-aware
//! predicate evaluator that gates skills on the active graph's shape.

use anyhow::{bail, Result};
use mcp_methods::server::{
    serve_prompts, BundledSkill, Manifest, McpServer, PredicateClause, ServerOptions, SkillError,
    SkillPredicateEvaluator, SkillRegistry,
};

use crate::tools::GraphState;
use crate::*;

/// Client-side tool-discovery steer, folded into workspace-mode
/// `instructions` so every `--workspace` / `workspace.kind: local`
/// deployment emits it on `initialize` without copy-pasting it into each
/// manifest. It complements the 0.12.6 in-band steering (graph-over-grep
/// vocabulary in tool descriptions, the activation mini-map, the result
/// footer) by making the *"search the registry"* instruction explicit for
/// lazy-tool-discovery clients (Codex / code_mode / tool-search), which can
/// surface only `grep`/`read_source` on a broad first query and miss the
/// always-registered graph tools. Skipped when the manifest already carries
/// equivalent guidance (see the dedup check in `run_async`).
pub(crate) const DISCOVERY_STEER: &str = "Tool discovery: graph_overview and cypher_query are ALWAYS registered. \
If a broad first tool-search surfaces only grep/read_source, search your tool registry for 'cypher' or \
'graph_overview' and load those before falling back to grep — a discovery miss does not mean the graph \
path is unavailable.";

pub(crate) const RECIPE_QUERIES_SKILL: &str = include_str!("../skills/recipe_queries.md");

/// Compose and serve the skill registry for a manifest-backed deployment.
///
/// Bundled methodology for KGLite's custom tools, the optional recipe catalog,
/// and framework defaults is composed with the operator-side project layer and
/// any operator-declared domain skill packs. The predicate evaluator gates
/// `read_code_source` on `graph_has_node_type: [Function, Class]` so it stays
/// out of prompts/list when the active graph isn't a code-tree (legal-corpus /
/// o&g / etc. deployments). A registry that fails to build disables skills for
/// the session rather than failing boot — except for the one failure an
/// operator can fix by reading the message; see [`report_registry_failure`].
pub(crate) fn install_skills(
    server: &mut McpServer,
    manifest: &Manifest,
    graph_state: &GraphState,
    recipe_catalog_summary: Option<crate::recipe_queries::CatalogSummary>,
) -> Result<()> {
    // Skill `.md` bodies live at `crates/kglite-mcp-server/skills/` — the
    // single canonical home. `cargo publish` only packages files inside
    // the crate dir, so they must live here (not behind a
    // `../../../kglite/...` `include_str!` path).
    let registry = SkillRegistry::new()
        .add_bundled(BundledSkill {
            name: "cypher_query",
            body: include_str!("../skills/cypher_query.md"),
        })
        .add_bundled(BundledSkill {
            name: "graph_overview",
            body: include_str!("../skills/graph_overview.md"),
        })
        .add_bundled(BundledSkill {
            name: "save_graph",
            body: include_str!("../skills/save_graph.md"),
        })
        .add_bundled(BundledSkill {
            name: "read_code_source",
            body: include_str!("../skills/read_code_source.md"),
        })
        .add_bundled(BundledSkill {
            name: "explore",
            body: include_str!("../skills/explore.md"),
        })
        // Cross-tool skills: named after no tool, they attach via
        // `references_tools` and lead with the `description` routing —
        // both rely on the serve_prompts injection added in mcp-methods
        // 0.3.42 (## When to use + references_tools), so they only became
        // active with that pin bump.
        .add_bundled(BundledSkill {
            name: "code_graph_analysis",
            body: include_str!("../skills/code_graph_analysis.md"),
        })
        .add_bundled(BundledSkill {
            name: "code_graph_views",
            body: include_str!("../skills/code_graph_views.md"),
        });
    let registry_result = add_recipe_query_skill(registry, recipe_catalog_summary)
        .merge_framework_defaults()
        .auto_detect_project_layer(&manifest.yaml_path)
        .layer_dirs(&manifest.skills, &manifest.yaml_path)
        .and_then(|r| {
            r.with_predicate_evaluator(KglitePredicateEvaluator {
                state: graph_state.clone(),
            })
            .finalise()
        });
    match registry_result {
        Ok(registry) => {
            serve_prompts(&registry, server);
            Ok(())
        }
        Err(e) => report_registry_failure(e, &manifest.yaml_path),
    }
}

/// Decide what a failed registry build costs: the boot, or a warning.
///
/// A declared pack directory that is not there is an operator typo, and one
/// bad entry fails the *whole* build — the bundled methodology goes with it.
/// Nothing else in the session says so: the graph tools still answer, and
/// `--selftest` used to print PASSED over a server with every skill silently
/// gone. `source_root:` reports its own version of this state; skills had no
/// equivalent, so a missing pack refuses the boot and names both spellings of
/// the path.
///
/// Everything else — an unparseable skill file, one over the size limit — stays
/// a warning: those are content faults in files that do exist, they name
/// themselves in the log, and taking a deployment down for one of them is a
/// worse trade than serving it without skills.
fn report_registry_failure(error: SkillError, yaml_path: &std::path::Path) -> Result<()> {
    match error {
        SkillError::PathNotFound { .. } => bail!(
            "{error}. Declared by `skills:` in {}. Create the directory, or drop the entry \
             — a skills path that is not there disables every skill in the session, \
             bundled ones included.",
            yaml_path.display()
        ),
        other => {
            tracing::warn!(error = %other, "skills registry build failed; skills disabled for this session");
            Ok(())
        }
    }
}

/// Add recipe methodology only when the validated catalog will register its
/// fixed routes. The skill's `tool_registered: run_recipe_query` predicate is
/// a second guard evaluated against the final visible tool set.
pub(crate) fn add_recipe_query_skill(
    registry: SkillRegistry,
    catalog_summary: Option<recipe_queries::CatalogSummary>,
) -> SkillRegistry {
    if catalog_summary.is_some() {
        registry.add_bundled(BundledSkill {
            name: "recipe_queries",
            body: RECIPE_QUERIES_SKILL,
        })
    } else {
        registry
    }
}

/// Fold [`DISCOVERY_STEER`] into `options.instructions` for the two
/// workspace modes. Appends (preserving any manifest `instructions:`) or
/// sets it when none exists; bails when the text already mentions the
/// always-registered graph tools so an opted-in manifest isn't duplicated.
pub(crate) fn apply_discovery_steer(mode: &Mode, mut options: ServerOptions) -> ServerOptions {
    if !matches!(mode, Mode::Workspace { .. } | Mode::LocalWorkspace { .. }) {
        return options;
    }
    let already = options
        .instructions
        .as_deref()
        .is_some_and(|s| s.to_lowercase().contains("always registered"));
    if already {
        return options;
    }
    options.instructions = Some(match options.instructions.take() {
        Some(existing) if !existing.trim().is_empty() => format!("{existing}\n\n{DISCOVERY_STEER}"),
        _ => DISCOVERY_STEER.to_string(),
    });
    options
}

/// Evaluates `applies_when:` predicates that depend on kglite's
/// runtime graph state. The framework dispatches `tool_registered:`
/// and `extension_enabled:` itself; this evaluator only handles the
/// two domain predicates that require knowing what node types and
/// properties the active graph carries.
///
/// Returning `None` for an unrecognised clause marks the predicate
/// `Unknown` upstream — the framework's safe default suppresses the
/// skill when any clause is `Unknown`, which prevents a typo'd
/// predicate from silently activating a skill against the wrong
/// domain.
pub(crate) struct KglitePredicateEvaluator {
    pub(crate) state: GraphState,
}

impl SkillPredicateEvaluator for KglitePredicateEvaluator {
    fn evaluate(&self, clause: &PredicateClause<'_>) -> Option<bool> {
        match clause {
            PredicateClause::GraphHasNodeType(types) => {
                Some(types.iter().any(|t| self.state.has_node_type(t)))
            }
            PredicateClause::GraphHasProperty {
                node_type,
                prop_name,
            } => Some(self.state.has_property(node_type, prop_name)),
            // Framework-internal predicates — `tool_registered` and
            // `extension_enabled` are dispatched against ServerOptions
            // by the framework itself, not via this evaluator.
            _ => None,
        }
    }
}

#[cfg(test)]
mod registry_failure_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn a_missing_declared_pack_refuses_the_boot_and_names_both_spellings() {
        // Both spellings, because the operator wrote one and the filesystem
        // holds the other: a `./domain` that resolved somewhere unexpected is
        // the same typo as a `./domian` that resolved exactly where asked.
        let error = SkillError::PathNotFound {
            raw: "./domain".to_string(),
            resolved: PathBuf::from("/srv/deploy/domain"),
        };

        let message = report_registry_failure(error, Path::new("/srv/deploy/graph_mcp.yaml"))
            .expect_err("a declared pack that is not there must fail boot")
            .to_string();

        assert!(message.contains("./domain"), "{message}");
        assert!(message.contains("/srv/deploy/domain"), "{message}");
        assert!(message.contains("/srv/deploy/graph_mcp.yaml"), "{message}");
    }

    #[test]
    fn a_content_fault_in_a_file_that_exists_stays_a_warning() {
        // The file is there and the log names it; refusing the whole
        // deployment over one malformed skill is the worse trade.
        let error = SkillError::MissingFrontmatter {
            path: PathBuf::from("/srv/deploy/domain/broken.md"),
        };

        assert!(report_registry_failure(error, Path::new("/srv/deploy/graph_mcp.yaml")).is_ok());
    }
}

#[cfg(test)]
mod discovery_steer_tests {
    use super::*;
    use std::path::PathBuf;

    use mcp_methods::server::ServerOptions;

    fn ws_mode() -> Mode {
        Mode::LocalWorkspace {
            root: PathBuf::from("/tmp/ws"),
            watch: false,
        }
    }

    #[test]
    fn appends_to_workspace_modes() {
        let out = apply_discovery_steer(&ws_mode(), ServerOptions::default());
        let text = out.instructions.expect("instructions set");
        assert!(text.contains("ALWAYS registered"));
        assert!(text.contains("cypher"));
    }

    #[test]
    fn preserves_manifest_instructions() {
        let opts = ServerOptions {
            instructions: Some("Domain guidance here.".to_string()),
            ..Default::default()
        };
        let out = apply_discovery_steer(&ws_mode(), opts);
        let text = out.instructions.expect("instructions set");
        assert!(text.starts_with("Domain guidance here."));
        assert!(text.contains("ALWAYS registered"));
    }

    #[test]
    fn dedupes_when_already_present() {
        let opts = ServerOptions {
            instructions: Some(
                "graph_overview and cypher_query are ALWAYS registered.".to_string(),
            ),
            ..Default::default()
        };
        let out = apply_discovery_steer(&ws_mode(), opts);
        let text = out.instructions.expect("instructions set");
        // Only the manifest's own copy — not appended a second time.
        assert_eq!(text.matches("ALWAYS registered").count(), 1);
    }

    #[test]
    fn skips_non_workspace_modes() {
        let mode = Mode::Graph {
            path: PathBuf::from("/tmp/g.kgl"),
        };
        let out = apply_discovery_steer(&mode, ServerOptions::default());
        assert!(out.instructions.is_none());
    }
}

#[cfg(test)]
mod recipe_skill_tests {
    use super::*;

    use mcp_methods::server::{serve_prompts, McpServer, ServerOptions, SkillRegistry};

    use mcp_methods::server::{SkillSource, SkillsSource};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Default, Deserialize, JsonSchema)]
    struct EmptyArgs {}

    fn catalog(raw: Option<&serde_json::Value>) -> recipe_queries::RecipeCatalog {
        recipe_queries::RecipeCatalog::from_manifest_value(raw).expect("valid catalog")
    }

    fn present_catalog() -> recipe_queries::RecipeCatalog {
        catalog(Some(&json!({
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
    }

    fn resolved_recipe_skills(
        summary: Option<recipe_queries::CatalogSummary>,
        source: &SkillsSource,
    ) -> mcp_methods::server::ResolvedRegistry {
        add_recipe_query_skill(SkillRegistry::new(), summary)
            .layer_dirs(source, std::path::Path::new("manifest.yaml"))
            .expect("configure skill layers")
            .finalise()
            .expect("resolve recipe skill")
    }

    fn bundled_source() -> SkillsSource {
        SkillsSource::Sources(vec![SkillSource::Bundled])
    }

    #[test]
    fn recipe_skill_is_conditionally_bundled_for_present_catalog_only() {
        let source = bundled_source();
        let absent = catalog(None);
        let empty_value = json!({});
        let empty = catalog(Some(&empty_value));
        let present = present_catalog();

        for summary in [absent.discovery_summary(), empty.discovery_summary()] {
            let registry = resolved_recipe_skills(summary, &source);
            assert!(!registry
                .skill_names()
                .iter()
                .any(|name| name == "recipe_queries"));
        }
        let registry = resolved_recipe_skills(present.discovery_summary(), &source);
        assert!(registry
            .skill_names()
            .iter()
            .any(|name| name == "recipe_queries"));

        let disabled = resolved_recipe_skills(present.discovery_summary(), &SkillsSource::Disabled);
        assert!(
            disabled.is_empty(),
            "a present catalog must not bypass the manifest skills opt-in"
        );
    }

    #[test]
    fn recipe_skill_contract_names_direct_discovery_preflight_and_raw_fallback() {
        let source = bundled_source();
        let registry = resolved_recipe_skills(present_catalog().discovery_summary(), &source);
        let skill = registry.get("recipe_queries").expect("recipe skill");

        assert_eq!(
            skill.frontmatter.references_tools,
            ["list_recipe_queries", "run_recipe_query", "cypher_query"]
        );
        let applies = skill
            .frontmatter
            .applies_when
            .as_ref()
            .expect("tool registration gate");
        assert_eq!(
            applies.tool_registered.as_deref(),
            Some(recipe_queries::RUN_RECIPE_QUERY_TOOL)
        );
        assert_eq!(applies.extension_enabled, None);
        let body = skill.body.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(body.contains("Do not call `list_recipe_queries` first"));
        assert!(body.contains("domain skill already selected the"));
        assert!(body.contains("call `list_recipe_queries()` once"));
        assert!(body.contains("exactly matches the requested scope"));
        assert!(body.contains("mandatory `resolve_*` preflight"));
        assert!(body.contains("fall back to raw `cypher_query`"));
    }

    #[test]
    fn tool_registered_gate_beats_truthy_empty_extension_and_injects_all_references() {
        let source = bundled_source();
        let registry = resolved_recipe_skills(present_catalog().discovery_summary(), &source);
        let mut without_route = McpServer::new(ServerOptions {
            extensions: serde_json::Map::from_iter([("cypher_recipes".to_string(), json!({}))]),
            ..ServerOptions::default()
        });
        serve_prompts(&registry, &mut without_route);
        assert!(
            without_route
                .prompt_router_mut()
                .list_all()
                .iter()
                .all(|prompt| prompt.name != "recipe_queries"),
            "an empty extension mapping is truthy but must not activate the skill"
        );

        let registry = resolved_recipe_skills(present_catalog().discovery_summary(), &source);
        let mut with_routes = McpServer::new(ServerOptions::default());
        for name in [
            recipe_queries::LIST_RECIPE_QUERIES_TOOL,
            recipe_queries::RUN_RECIPE_QUERY_TOOL,
            "cypher_query",
        ] {
            with_routes.register_typed_tool::<EmptyArgs, _>(name, "base", |_| "ok".to_string());
        }
        serve_prompts(&registry, &mut with_routes);

        assert!(with_routes
            .prompt_router_mut()
            .list_all()
            .iter()
            .any(|prompt| prompt.name == "recipe_queries"));
        for name in [
            recipe_queries::LIST_RECIPE_QUERIES_TOOL,
            recipe_queries::RUN_RECIPE_QUERY_TOOL,
            "cypher_query",
        ] {
            let description = with_routes
                .tool_router_mut()
                .get(name)
                .and_then(|tool| tool.description.as_deref())
                .expect("tool description");
            assert!(
                description.contains("mcp-skill:recipe_queries"),
                "{name} missing recipe methodology injection"
            );
        }
    }
}

#[cfg(test)]
mod bundled_skill_body_tests {
    /// The generic `cypher_query` skill ships to every deployment — shipping,
    /// legal, maritime — with no `applies_when` gate, because Cypher applies
    /// to any graph. Code-graph *methodology* does not: it opened with the
    /// four-step code-graph workflow and "Never `grep` for a definition",
    /// ~1.5k tokens of instruction about a codebase, delivered verbatim to a
    /// graph of vessels. That content already lives in `code_graph_analysis`,
    /// which gates on `graph_has_node_type: [Function, Class]` and reaches
    /// the readers it is for.
    #[test]
    fn the_generic_cypher_skill_carries_no_code_graph_preamble() {
        let body = include_str!("../skills/cypher_query.md");
        for marker in [
            "Never `grep`",
            "Code-graph workflow",
            "read_code_source(qualified_name=…)",
        ] {
            assert!(
                !body.contains(marker),
                "cypher_query.md still carries code-graph methodology: {marker:?}"
            );
        }
        // The gated skill is where it belongs, and still has it.
        let gated = include_str!("../skills/code_graph_analysis.md");
        assert!(gated.contains("Never `grep`"));
        assert!(gated.contains("Code-graph workflow"));
    }

    /// Two things the skill previously got wrong about its own tool: it said
    /// `$name` parameters "aren't currently exposed" (they are, via `params`),
    /// and it presented `FORMAT CSV` as the way to get a large result out
    /// while the inline body is capped at 200 rows.
    #[test]
    fn the_generic_cypher_skill_states_the_current_tool_contract() {
        let body = include_str!("../skills/cypher_query.md");
        assert!(!body.contains("aren't currently exposed"));
        assert!(body.contains("params="));
        assert!(body.contains("200 data rows"));
        assert!(body.contains("openCypher"));
    }
}
