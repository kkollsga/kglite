//! The async server body: one named boot phase after another, ending in
//! the stdio MCP service.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use mcp_methods::server::{init_tracing, load_env_for_mode, McpServer, ServerOptions};

use crate::tools::GraphState;
use crate::*;

/// Every `extensions:` key this crate reads. mcp-methods stores the block as an
/// unvalidated map (`Manifest.extensions`), so nothing upstream notices a
/// misspelled key — and the failure direction of a typo is silent: the feature
/// is simply off. **A new `boot_*` extension reader adds its key here in the
/// same change**, or [`unknown_extension_keys`] reports the working key as a
/// typo forever after.
const KNOWN_EXTENSION_KEYS: &[&str] = &[
    "cypher_recipes",
    "recipe_catalog",
    "recipe_tools",
    "value_codecs",
    "ontology",
    // Retired but still recognised — see `boot_graph_watch`.
    "graph_watch",
    "parallel",
    "tools_allow",
    "write_scope",
    "csv_http_server",
    "embedder",
    "writable",
    "fetch_images",
    "valid_time",
    "embeddings",
    "columns",
    "topology",
    "write_budget",
];

/// Manifest `extensions:` keys this build does not read, for the boot warning.
///
/// A warning rather than a boot error, and deliberately: a skill's
/// `applies_when: { extension_enabled: <key> }` predicate reads the same block
/// and legitimately names a key no `boot_*` reader knows, so refusing the boot
/// would break a working deployment. Returned rather than logged so the set is
/// testable without a tracing subscriber; the call site does the logging.
fn unknown_extension_keys(manifest: Option<&mcp_methods::server::Manifest>) -> Vec<String> {
    manifest
        .into_iter()
        .flat_map(|m| m.extensions.keys())
        .filter(|key| !KNOWN_EXTENSION_KEYS.contains(&key.as_str()))
        .cloned()
        .collect()
}

/// Compile and validate the manifest's stored Cypher recipes once at boot.
///
/// Recipe definitions are configuration, not per-call input, so the
/// deliberately small schema subset is checked here and later route handlers
/// share one immutable catalog.
/// The manifest's catalogue alone. The graph's own records are merged under it
/// later ([`recipe_queries::merge_graph_recipes`]), once the graph is open.
fn boot_recipe_catalog(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<recipe_queries::RecipeCatalog> {
    let catalog = match manifest {
        Some(manifest) => recipe_queries::RecipeCatalog::from_manifest_value(
            manifest.extensions.get("cypher_recipes"),
        )
        .context("extensions.cypher_recipes parse failed")?,
        None => recipe_queries::RecipeCatalog::default(),
    };
    if let Some(summary) = catalog.discovery_summary() {
        tracing::info!(
            recipes = summary.recipe_count,
            queries = summary.query_count,
            "Cypher recipe catalog validated"
        );
    }
    Ok(catalog)
}

/// Build the manifest-declared, position-scoped literal codecs
/// (`extensions.value_codecs`: prefix / map / regex).
///
/// Passed to the engine via `ExecuteOptions` per query — decode on the way in,
/// encode on the way out. Empty when absent; a malformed block errors at boot,
/// not per-query.
fn boot_value_codecs(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Arc<Vec<kglite::api::cypher::ValueCodec>>>> {
    let codecs = match manifest {
        Some(m) => value_codecs::from_manifest(m.extensions.get("value_codecs"))
            .context("extensions.value_codecs parse failed")?,
        None => Vec::new(),
    };
    Ok((!codecs.is_empty()).then(|| Arc::new(codecs)))
}

use crate::tools::BoundOntology;

/// `extensions.ontology: {"file": "x.json"}` — the declared semantic layer,
/// parsed once at boot (a malformed value fails the server, per the crate's
/// no-silently-ignored-keys rule) and applied memory-only to every graph the
/// state installs. The path resolves against the manifest's directory, like
/// `csv_http_server`.
fn boot_ontology(
    manifest: Option<&mcp_methods::server::Manifest>,
    manifest_base: &Path,
) -> Result<Option<BoundOntology>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("ontology")) else {
        return Ok(None);
    };
    let materialize = match raw.get("materialize") {
        None => false,
        Some(v) => v
            .as_bool()
            .context("extensions.ontology.materialize must be a boolean")?,
    };
    let file = raw
        .get("file")
        .and_then(|v| v.as_str())
        .context("extensions.ontology must be a map with a 'file' path (JSON document)")?;
    let path = if Path::new(file).is_absolute() {
        PathBuf::from(file)
    } else {
        manifest_base.join(file)
    };
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("extensions.ontology: cannot read {}", path.display()))?;
    let store = kglite::api::ontology_from_json(&text)
        .map_err(|e| anyhow::anyhow!("extensions.ontology {}: {e}", path.display()))?;
    tracing::info!(
        classes = store.classes.len(),
        relationships = store.relationships.len(),
        materialize,
        "manifest ontology parsed (memory-only; persists only via an explicit save_graph)"
    );
    Ok(Some(BoundOntology {
        store: Arc::new(store),
        materialize,
    }))
}

/// `extensions.graph_watch` — **retired**, and still parsed.
///
/// What it used to arm is now unconditional: a `--graph` server stats the
/// served file on every tool call (`tools::graph_reload`), so there is no
/// watcher to switch on and nothing an operator gains by setting the key.
/// Returning what was parsed rather than dropping it is what lets the caller
/// warn about a key that no longer does anything — a manifest carrying
/// `graph_watch: false` used to *mean* "do not refresh", and its author has to
/// learn that it no longer says that. A non-boolean value is still a boot
/// error rather than a silently ignored key (see [`boot_tools_allow`]): the
/// day this key is finally deleted, a typo in it must not start passing.
fn boot_graph_watch(manifest: Option<&mcp_methods::server::Manifest>) -> Result<Option<bool>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("graph_watch")) else {
        return Ok(None);
    };
    raw.as_bool()
        .map(Some)
        .context("extensions.graph_watch must be a boolean (true or false)")
}

/// `extensions.parallel: true` — the manifest half of the parallel opt-in.
///
/// Boolean-key handling matches [`boot_tools_allow`]: absent is off, a
/// non-boolean is a boot error rather than a silently dropped key.
fn boot_manifest_parallel(manifest: Option<&mcp_methods::server::Manifest>) -> Result<bool> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("parallel")) else {
        return Ok(false);
    };
    raw.as_bool()
        .context("extensions.parallel must be a boolean (true or false)")
}

/// Resolve the parallel-runtime opt-in from `--parallel` and
/// `extensions.parallel`.
///
/// **OR, not intersection** — the opposite of [`boot_write_scope`], and for
/// the reason that makes the two opposite: a write scope is a *perimeter*, so
/// combining two of them may only narrow, while this is a *resource
/// permission* whose worst case is a query using more cores than it needed.
/// Either surface alone turns it on, so a wrapper that owns the manifest but
/// not argv, and a bare binary launched with no manifest at all, each have a
/// working way to say yes.
///
/// Applies to the read seam only ([`tools::ExecPolicy`]); mutations stay
/// sequential.
fn boot_parallel(manifest: Option<&mcp_methods::server::Manifest>, cli: &Cli) -> Result<bool> {
    Ok(boot_manifest_parallel(manifest)? || cli.parallel)
}

/// The graph's valid-time default for this server: `--valid-time-default`
/// when given, else `extensions.valid_time.default`, else `None` (graphs keep
/// the engine's `today`). The flag wins outright rather than combining: a
/// wrapper that owns argv overrides the manifest it ships with. A malformed
/// manifest value, or a key other than `default`, fails the boot rather than
/// being silently ignored.
fn boot_valid_time_default(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<kglite::api::temporal::ValidTimeDefault>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("valid_time")) else {
        return Ok(cli.valid_time_default);
    };
    let map = raw
        .as_object()
        .context("extensions.valid_time must be a map with an optional 'default' key")?;
    if let Some(key) = map.keys().find(|key| key.as_str() != "default") {
        anyhow::bail!("extensions.valid_time: unknown key '{key}'; the only key is 'default'");
    }
    let configured = match map.get("default") {
        None => None,
        Some(value) => {
            let text = value.as_str().context(
                "extensions.valid_time.default must be 'today', 'all' or a YYYY-MM-DD date",
            )?;
            Some(
                kglite::api::temporal::ValidTimeDefault::parse(text)
                    .map_err(|e| anyhow::anyhow!("extensions.valid_time.default: {e}"))?,
            )
        }
    };
    Ok(cli.valid_time_default.or(configured))
}

/// The placement for one of the three mappable data classes (`embeddings`,
/// `columns` or `topology`) of every `.kgl` the server opens: the flag when given, else
/// `extensions.<key>.storage`, else `None` (the engine's default, `auto`). The
/// flag wins outright, like `--valid-time-default`. A malformed value, or a key
/// other than `storage`, fails the boot rather than being silently ignored.
fn boot_map_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    key: &str,
    flag: Option<kglite::api::io::MapPolicy>,
) -> Result<Option<kglite::api::io::MapPolicy>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get(key)) else {
        return Ok(flag);
    };
    let map = raw.as_object().with_context(|| {
        format!("extensions.{key} must be a map with an optional 'storage' key")
    })?;
    if let Some(unknown) = map.keys().find(|name| name.as_str() != "storage") {
        anyhow::bail!("extensions.{key}: unknown key '{unknown}'; the only key is 'storage'");
    }
    let configured = match map.get("storage") {
        None => None,
        Some(value) => {
            let text = value.as_str().with_context(|| {
                format!("extensions.{key}.storage must be 'auto', 'mapped' or 'memory'")
            })?;
            Some(
                kglite::api::io::MapPolicy::from_name(text).with_context(|| {
                    format!("extensions.{key}.storage: '{text}' is not one of auto, mapped, memory")
                })?,
            )
        }
    };
    Ok(flag.or(configured))
}

/// The embedding placement: `--embeddings`, else `extensions.embeddings.storage`.
fn boot_embeddings_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<kglite::api::io::MapPolicy>> {
    boot_map_policy(manifest, "embeddings", cli.embeddings)
}

/// The property-column placement: `--columns`, else `extensions.columns.storage`.
fn boot_columns_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<kglite::api::io::MapPolicy>> {
    boot_map_policy(manifest, "columns", cli.columns)
}

/// The topology and index placement: `--topology`, else `extensions.topology.storage`.
fn boot_topology_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<kglite::api::io::MapPolicy>> {
    boot_map_policy(manifest, "topology", cli.topology)
}

/// The unsaved-write budget of every `.kgl` the server opens: `--write-budget`,
/// else `extensions.write_budget.limit`, else `None` (the engine's default).
fn boot_write_budget(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<kglite::api::io::WriteBudgetSetting>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("write_budget")) else {
        return Ok(cli.write_budget);
    };
    let map = raw
        .as_object()
        .context("extensions.write_budget must be a map with an optional 'limit' key")?;
    if let Some(unknown) = map.keys().find(|name| name.as_str() != "limit") {
        anyhow::bail!("extensions.write_budget: unknown key '{unknown}'; the only key is 'limit'");
    }
    let configured = match map.get("limit") {
        None => None,
        Some(value) => {
            let text = match value {
                serde_json::Value::String(text) => text.clone(),
                serde_json::Value::Number(number) => number.to_string(),
                _ => anyhow::bail!(
                    "extensions.write_budget.limit must be 'default', 'unlimited' or a size"
                ),
            };
            Some(
                kglite::api::io::WriteBudgetSetting::parse(&text).map_err(|message| {
                    anyhow::anyhow!("extensions.write_budget.limit: {message}")
                })?,
            )
        }
    };
    Ok(cli.write_budget.or(configured))
}

/// `extensions.writable: true` — the manifest half of the write opt-in.
///
/// Boolean-key handling matches [`boot_manifest_parallel`]: absent is off, a
/// non-boolean is a boot error rather than a silently dropped key. That
/// strictness is what keeps the typo's failure direction safe *and* loud —
/// a mistyped **value** fails the boot, and a mistyped **key** leaves a
/// read-only server that warns about the unknown key
/// ([`unknown_extension_keys`]) and names both spellings when it refuses the
/// first mutation (`tools::MUTATION_NOT_ALLOWED`).
fn boot_manifest_writable(manifest: Option<&mcp_methods::server::Manifest>) -> Result<bool> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("writable")) else {
        return Ok(false);
    };
    raw.as_bool()
        .context("extensions.writable must be a boolean (true or false)")
}

/// Whether agents may mutate the served graph: `--writable` or
/// `extensions.writable: true`, unless the embedder pinned
/// [`ServerExtensions::read_only`].
///
/// **OR, not intersection**, for [`boot_parallel`]'s reason: a wrapper that
/// owns the manifest but not argv, and a bare binary launched with no manifest
/// at all, each need a working way to say yes. The pin is the one veto, and it
/// comes from the binary embedding this server — argv and the manifest are
/// both operator surfaces, the pin is the embedder's own contract.
///
/// The two opt-in surfaces are read *separately* rather than OR-ed into one
/// answer, because a pin that silently swallows them leaves the operator who
/// set one staring at a read-only server with nothing in the log naming the
/// key they typed.
///
/// Resolved once at boot and threaded as a `bool` to every consumer — the
/// write-scope warning, [`writer_lease_policy`], [`boot_builtins`] and the
/// code-tool gate — so no consumer can re-derive it and disagree. Distinct
/// from [`owns_graph_file`]: this is "the agent may write", that is "this
/// server may rewrite the file".
fn boot_mutations_enabled(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
    read_only_pin: bool,
) -> Result<bool> {
    let from_manifest = boot_manifest_writable(manifest)?;
    let from_cli = cli.writable;
    if read_only_pin && (from_manifest || from_cli) {
        let overridden = match (from_manifest, from_cli) {
            (true, true) => "extensions.writable: true and --writable",
            (true, false) => "extensions.writable: true",
            _ => "--writable",
        };
        tracing::warn!(
            overridden,
            "this build pins the server read-only, so the write opt-in is inert; mutations              stay refused"
        );
    }
    Ok(!read_only_pin && (from_manifest || from_cli))
}

/// Fallback name for the lease record: the process that spawned this server.
///
/// Best-effort by construction — every failure returns `None` and the caller
/// falls back to the binary's own name. Zero new dependencies: `/proc` on
/// Linux, one `ps` on macOS, nothing on Windows (where the owner record still
/// carries pid and timestamp).
#[cfg(unix)]
fn parent_process_name() -> Option<String> {
    let ppid = std::os::unix::process::parent_id();
    #[cfg(target_os = "linux")]
    let raw = std::fs::read_to_string(format!("/proc/{ppid}/comm")).ok()?;
    #[cfg(not(target_os = "linux"))]
    let raw = {
        let output = std::process::Command::new("ps")
            .args(["-o", "comm=", "-p", &ppid.to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout).ok()?
    };
    // macOS `ps -o comm=` reports the executable path, not the basename.
    let name = raw.trim().rsplit('/').next()?.trim();
    (!name.is_empty()).then(|| name.to_owned())
}

#[cfg(not(unix))]
fn parent_process_name() -> Option<String> {
    None
}

/// Resolve the name this server publishes in the graph's `<path>.lock-owner`
/// record: `--lease-label`, else `KGLITE_LEASE_LABEL`, else the parent process
/// name, else the binary's own name.
///
/// Mirrors [`boot_parallel`]'s shape — resolved once at boot, carried by every
/// clone of the state. There is deliberately **no manifest key**: the case this
/// exists for is four MCP clients sharing one manifest on one graph, so a
/// manifest-declared label would give all four the same name and answer
/// nothing. The env var is read explicitly rather than through clap's `env`
/// attribute so the precedence above is visible here rather than in an
/// attribute two files away.
fn boot_lease_label(cli: &Cli) -> String {
    if let Some(label) = cli.lease_label.as_deref().map(str::trim) {
        if !label.is_empty() {
            return label.to_owned();
        }
    }
    if let Ok(label) = std::env::var("KGLITE_LEASE_LABEL") {
        let label = label.trim();
        if !label.is_empty() {
            return label.to_owned();
        }
    }
    parent_process_name().unwrap_or_else(|| "kglite-mcp-server".to_owned())
}

/// The width the engine's query pool will be built at, for the boot log.
///
/// Recomputed here rather than read from the engine: the pool is built lazily
/// on the first query that crosses a fan-out threshold, so at boot there is
/// nothing to ask. Mirrors `kglite::graph::parallel::configured_width` — same
/// env var, same `available_parallelism` fallback — and is reported as
/// observed configuration, never as a promise about a pool that does not
/// exist yet.
fn query_pool_width() -> (usize, bool) {
    let override_width = std::env::var_os("KGLITE_QUERY_THREADS")
        .and_then(|raw| raw.to_str()?.trim().parse::<usize>().ok())
        .filter(|n| *n > 0);
    match override_width {
        Some(n) => (n, true),
        None => (
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            false,
        ),
    }
}

/// `extensions.tools_allow: [name, ...]` — the closed-by-default tool surface.
///
/// Absent key = no allowlist = every registered route stays visible. Present,
/// it names the *final* tool surface: see [`apply_tool_allowlist`] for what
/// that means and what it deliberately does not do. An explicit empty list is
/// honoured literally — a server with no tools — because the alternative
/// (silently treating `[]` as "no allowlist") would widen a surface an
/// operator asked to close.
///
/// A non-list value, or a non-string element, is a boot error rather than a
/// dropped key: an allowlist that silently fails open is worse than no
/// allowlist, and "parsed and then ignored" is this crate's recurring defect
/// shape (`temp_cleanup`).
/// `extensions.fetch_images: { max_items, max_bytes_per_image, max_total_bytes }`.
///
/// Absent block = the defaults (4 images, 4 MiB each, 12 MiB total). Read at
/// registration rather than in [`boot_graph`] because the caps are the route's
/// own, and the route is where a malformed block should stop the boot.
/// `extensions.recipe_catalog: { block_budget, description_budget }`.
///
/// Absent block = the defaults. Read here rather than inside the catalogue
/// because the budgets are the *description's*, not the catalogue's: they
/// change what `tools/list` publishes about a catalogue that compiled either
/// way.
fn boot_recipe_catalog_budgets(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<recipe_queries::CatalogBudgets> {
    recipe_queries::CatalogBudgets::from_manifest_value(
        manifest.and_then(|m| m.extensions.get("recipe_catalog")),
    )
}

/// `extensions.recipe_tools: false` — the operator's off switch for the
/// per-query named routes a catalogue's `tool:` keys ask for.
///
/// Default on, the same posture as producer skills: the catalogue author
/// curates which queries declare a name, each one runs the same validated
/// read-only Cypher `run_recipe_query` does, and a name already registered
/// refuses the boot. A non-boolean is a boot error, not a dropped key.
fn boot_recipe_tools_enabled(manifest: Option<&mcp_methods::server::Manifest>) -> Result<bool> {
    match manifest.and_then(|m| m.extensions.get("recipe_tools")) {
        None => Ok(true),
        Some(raw) => raw
            .as_bool()
            .context("extensions.recipe_tools must be a boolean"),
    }
}

/// Both catalogue-publishing knobs, read together because both decide what
/// `register_recipe_query_routes` puts in `tools/list`.
fn boot_recipe_routes(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<recipe_queries::RecipeRouteOptions> {
    Ok(recipe_queries::RecipeRouteOptions {
        budgets: boot_recipe_catalog_budgets(manifest)?,
        named_tools: boot_recipe_tools_enabled(manifest)?,
        manifest_tools: Vec::new(),
    })
}

/// Close the two code-graph routes on a graph that carries no code types.
///
/// Disable, never skip registration: an unregistered name is absent from
/// `router.map`, and `apply_bundled_tool_overrides` hard-errors on any
/// manifest override naming a route not in that map (only the
/// framework-gated `repo_management` gets a pass there). `disable_route`
/// keeps the entry — unlisted, rejected on call — while leaving overrides
/// resolvable.
fn disable_code_tools(server: &mut McpServer) {
    server.tool_router_mut().disable_route("explore");
    server.tool_router_mut().disable_route("read_code_source");
}

/// The catalogue's publishing options, and the tool names its `tool:` keys
/// will claim.
///
/// Both are read here rather than inside registration because the *names* are
/// needed twice: once by `register_recipe_query_routes`, and again by the
/// raw-stdio layer, which must capture the untouched JSON of every route that
/// binds Cypher parameters or a named recipe tool's integer guard never sees
/// the request it is guarding. Registration re-derives them and is the party
/// that refuses a duplicate, so a duplicate here is simply an empty list — the
/// boot fails moments later with the message that names both claimants.
fn recipe_route_plan(
    manifest: Option<&mcp_methods::server::Manifest>,
    catalog: &recipe_queries::RecipeCatalog,
) -> Result<(recipe_queries::RecipeRouteOptions, Vec<String>)> {
    let options = boot_recipe_routes(manifest)?;
    let names = if options.named_tools {
        catalog
            .tool_names()
            .map(|names| names.into_keys().collect())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    Ok((options, names))
}

fn boot_fetch_image_caps(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<fetch_images::ImageCaps> {
    fetch_images::ImageCaps::from_manifest_value(
        manifest.and_then(|m| m.extensions.get("fetch_images")),
    )
}

fn boot_tools_allow(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Vec<String>>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("tools_allow")) else {
        return Ok(None);
    };
    let items = raw
        .as_array()
        .context("extensions.tools_allow must be a list of tool names")?;
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let name = item.as_str().with_context(|| {
            format!("extensions.tools_allow entries must be tool names (strings); found {item}")
        })?;
        names.push(name.to_owned());
    }
    Ok(Some(names))
}

/// `extensions.write_scope: [NodeType, ...]` — the operator-pinned write scope.
///
/// Absent key = no pin = the agent's own `write_scope` argument decides (which
/// may be nothing at all). Present, it is the ceiling the agent can only narrow:
/// see [`resolve_write_scope`] for how the two combine. An explicit empty list
/// is honoured literally — a write-enabled server that permits no writes —
/// because reading `[]` as "no pin" would widen a perimeter the operator asked
/// to close.
///
/// A non-list value, or a non-string element, is a boot error rather than a
/// dropped key, for the same reason [`boot_tools_allow`] fails that way.
fn boot_manifest_write_scope(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Vec<String>>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("write_scope")) else {
        return Ok(None);
    };
    let items = raw
        .as_array()
        .context("extensions.write_scope must be a list of node types")?;
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let name = item.as_str().with_context(|| {
            format!("extensions.write_scope entries must be node types (strings); found {item}")
        })?;
        names.push(name.to_owned());
    }
    Ok(Some(names))
}

/// Resolve the operator-pinned write scope from the manifest key and the
/// `--write-scope` flag.
///
/// Both set is not an error: the two are **intersected**, the same rule the
/// agent's own scope obeys, so neither surface can widen what the other pinned.
/// Refusing the ambiguity at boot would only push the operator into deleting
/// one of them. The intersection is logged, so the effective scope shows in the
/// boot output rather than having to be inferred.
fn boot_write_scope(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
    mutations_enabled: bool,
) -> Result<Option<Vec<String>>> {
    let from_manifest = boot_manifest_write_scope(manifest)?;
    let from_flag = cli.write_scope.as_deref().map(parse_write_scope_flag);
    let resolved = match (from_manifest, from_flag) {
        (None, None) => None,
        (Some(only), None) | (None, Some(only)) => Some(only),
        (Some(m), Some(f)) => {
            let both: Vec<String> = f.iter().filter(|t| m.contains(t)).cloned().collect();
            tracing::info!(
                manifest = m.join(","),
                flag = f.join(","),
                effective = both.join(","),
                "write_scope pinned by both --write-scope and extensions.write_scope; \
                 the effective scope is their intersection"
            );
            Some(both)
        }
    };
    // `mutations_enabled`, not `owns_graph_file`: a manifest that only enables
    // `builtins.save_graph` registers `save_graph` but still serves the
    // read-only `cypher_query`, so the pin has nothing to apply to there
    // either.
    if resolved.is_some() && !mutations_enabled {
        tracing::warn!(
            "--write-scope / extensions.write_scope is set on a server whose cypher_query is \
             read-only; every mutation is refused already. Add --writable, or \
             extensions.writable: true, to serve scoped writes."
        );
    }
    Ok(resolved)
}

/// Directory the manifest was loaded from — the base both `csv_http_server`
/// (resolving `dir:`) and `temp_cleanup` (finding the directory to wipe)
/// resolve against. Falls back to cwd when there's no manifest.
fn manifest_base_dir(manifest: Option<&mcp_methods::server::Manifest>) -> PathBuf {
    manifest
        .and_then(|m| m.yaml_path.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Start the opt-in `extensions.csv_http_server:` CSV-over-HTTP listener.
///
/// When configured we spawn a tokio task to serve files out of the directory;
/// the `cypher_query` tool sees the same state and writes `FORMAT CSV` results
/// to that directory, returning a URL instead of an inline CSV blob.
///
/// `Err` is a malformed `csv_http_server` value only — a config-syntax error,
/// fatal like every other manifest mistake. A well-formed config whose
/// listener fails to *start* comes back as [`csv_http::CsvHttpState::Failed`]:
/// the extension is off, the boot continues, and both the summary line and
/// every `FORMAT CSV` answer say why.
async fn boot_csv_http(
    manifest: Option<&mcp_methods::server::Manifest>,
    manifest_base: &Path,
) -> Result<csv_http::CsvHttpState> {
    let cfg = match manifest.and_then(|m| m.extensions.get("csv_http_server")) {
        Some(raw) => csv_http::CsvHttpConfig::from_manifest_value(raw, manifest_base)
            .context("extensions.csv_http_server parse failed")?,
        None => None,
    };
    Ok(match cfg {
        Some(cfg) => csv_http::spawn(cfg).await,
        None => csv_http::CsvHttpState::Off,
    })
}

/// `builtins.save_graph` exactly as the manifest declared it, before
/// [`owns_graph_file`] widens it.
fn manifest_save_graph(manifest: Option<&mcp_methods::server::Manifest>) -> bool {
    manifest.map(|m| m.builtins.save_graph).unwrap_or(false)
}

/// Whether this deployment may **rewrite the graph file it serves** — a
/// manifest that enables `save_graph` on its own, or any write-enabled
/// deployment (an agent that mutates needs somewhere to persist, so
/// `save_graph` follows from `mutations_enabled`).
///
/// Read twice, once for each thing that follows from it: whether `save_graph`
/// is registered ([`tools::Builtins`]) and whether the server takes the path's
/// cross-process writer lease ([`writer_lease_policy`]). One function so those
/// two can never disagree about what "this server writes the file" means.
///
/// **Not** the mutation gate. `builtins.save_graph: true` alone owns the file —
/// it persists a boot-time ontology materialization — while `cypher_query`
/// stays read-only; that is [`boot_mutations_enabled`]'s question, and the two
/// predicates are separate names precisely because they answered as one for
/// long enough to be documented wrong.
fn owns_graph_file(save_graph: bool, mutations_enabled: bool) -> bool {
    save_graph || mutations_enabled
}

/// Whether to take the single-writer lease on the served path.
///
/// A read-only server never rewrites the file, so holding its lease refuses
/// every external rebuilder (`kglite.open(path)` fails fast) for the server's
/// whole lifetime and buys nothing back — and it stops two read-only servers
/// from serving one `.kgl`. Write-enabled deployments keep the exclusive lease;
/// so do disk-graph directories, whose retained mmaps make an external writer
/// genuinely unsafe rather than merely stale (see
/// `GraphState::takes_writer_lease`).
fn writer_lease_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    mutations_enabled: bool,
) -> tools::WriterLeasePolicy {
    if owns_graph_file(manifest_save_graph(manifest), mutations_enabled) {
        tools::WriterLeasePolicy::Exclusive
    } else {
        tools::WriterLeasePolicy::ReadOnly
    }
}

/// Builtin toggles from the manifest.
///   - `save_graph`: registration gated on [`owns_graph_file`], so a
///     destructive operation is not exposed to the agent on every graph
///     regardless of intent.
///   - `writable`: the mutation gate, resolved once by
///     [`boot_mutations_enabled`] from `--writable` and
///     `extensions.writable`.
///   - `temp_cleanup: on_overview`: wipe `temp/` on every bare
///     `graph_overview()`.
fn boot_builtins(
    manifest: Option<&mcp_methods::server::Manifest>,
    mutations_enabled: bool,
    csv_http: &csv_http::CsvHttpState,
    manifest_base: &Path,
    write_scope: Option<Vec<String>>,
) -> tools::Builtins {
    tools::Builtins {
        save_graph: owns_graph_file(manifest_save_graph(manifest), mutations_enabled),
        writable: mutations_enabled,
        write_scope,
        temp_cleanup_on_overview: manifest
            .map(|m| {
                matches!(
                    m.builtins.temp_cleanup,
                    mcp_methods::server::manifest::TempCleanup::OnOverview
                )
            })
            .unwrap_or(false),
        // Resolved against the manifest base, not the cwd — the server's cwd
        // need not match the manifest's parent. Reuse the csv_http_server
        // directory when configured, so both sides of the CSV pipeline agree
        // on what counts as "the temp dir".
        temp_dir: Some(
            csv_http
                .dir()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| manifest_base.join("temp")),
        ),
    }
}

/// Everything the kglite route block needs beyond the server it registers on.
///
/// A struct rather than a parameter list because the set grows with every boot
/// knob the overview decorations learn to carry, and each one that arrived as a
/// positional argument made the three call sites harder to read than the wiring
/// they describe.
struct KgliteToolParams<'a> {
    graph_state: &'a GraphState,
    manifest: Option<&'a mcp_methods::server::Manifest>,
    builtins: tools::Builtins,
    /// The catalogue actually served — the manifest's, merged with whatever
    /// the graph carried — as the bare-overview discovery hint.
    recipe_catalog_hint: Option<crate::recipe_queries::CatalogHint>,
    csv_http: Arc<csv_http::CsvHttpState>,
    source_roots_provider: Option<mcp_methods::server::source::SourceRootsProvider>,
    skills_index: tools::SkillsIndexSlot,
    /// Armed after `install_skills`; the graph-swap handlers hold a clone from
    /// registration time and drive the post-swap skill rebuild through it.
    skill_refresher: crate::skills::SkillRefresher,
}

fn register_kglite_tools(server: &mut McpServer, params: KgliteToolParams<'_>) -> Result<()> {
    let KgliteToolParams {
        graph_state,
        manifest,
        builtins,
        recipe_catalog_hint,
        csv_http,
        source_roots_provider,
        skills_index,
        skill_refresher,
    } = params;
    tools::register(
        server,
        graph_state.clone(),
        builtins,
        tools::OverviewDecorations {
            prefix: manifest.and_then(|manifest| manifest.overview_prefix.clone()),
            catalog: recipe_catalog_hint,
            skills: skills_index,
        },
        csv_http,
        skill_refresher,
    );
    code_source::register(server, graph_state.clone(), source_roots_provider.clone())
        .context("read_code_source registration failed")?;
    explore::register(server, graph_state.clone(), source_roots_provider.clone())
        .context("explore registration failed")?;
    // Registered in every mode, disabled where no root can serve the bytes —
    // an absent name would hard-fail any manifest override that mentions it.
    if let Some(reason) = fetch_images::register(
        server,
        source_roots_provider,
        boot_fetch_image_caps(manifest)?,
    )
    .context("fetch_images registration failed")?
    {
        tracing::info!(reason, "fetch_images registered but disabled");
    }
    Ok(())
}

/// Whether `explore` and `read_code_source` should be hidden for this boot.
///
/// `explore` pins its entry types and its traversal edge whitelist to code node
/// types, so on a graph without `Function`/`Class` it can only ever return "no
/// match" — it is structurally dead, and its presence in `tools/list` is pure
/// misdirection. Worse, `read_code_source`'s optional `node_type` argument makes
/// it a general-purpose reader of whatever `file_path` properties the graph
/// happens to carry, so on a non-code graph it is a way to pull arbitrary files
/// off disk with no code-graph purpose to justify it.
///
/// Three constraints are deliberate and settled:
///
/// 1. **This is a boot-time snapshot of a runtime property.** Every other
///    `has_node_type` consumer (skill `applies_when:` predicates, the result
///    footers) re-evaluates per call; this one cannot, because tool handlers
///    have no access to the router and so nothing can re-enable a route once
///    the server is running. A same-path graph reload rarely changes the
///    graph's *class* (a code graph stays a code graph), so the staleness
///    window is accepted. If handlers ever gain router access, make it live.
/// 2. **Write-enabled servers are exempt** — `--writable` or
///    `extensions.writable: true`, read here as `builtins.writable` (the
///    resolved `mutations_enabled`). `load_graph` can swap a code graph in
///    at any time, and by (1) there would be no way to re-enable the routes
///    afterwards — so such a server keeps both tools regardless of what
///    it booted with.
/// 3. **Graph mode only.** Every other mode has no graph at boot, so
///    `has_node_type` is uniformly `false` there and a mode-blind gate would
///    strip `explore` from exactly the deployments that need it most
///    (local-workspace and GitHub-repo code servers, whose graphs are built
///    after registration).
fn code_tools_are_dead(mode: &Mode, builtins: &tools::Builtins, graph_state: &GraphState) -> bool {
    matches!(mode, Mode::Graph { .. })
        && !builtins.writable
        && !(graph_state.has_node_type("Function") || graph_state.has_node_type("Class"))
}

/// Bind the manifest's `extensions.embedder:` as the embedder `text_score()`
/// uses.
///
/// Selection is by `library:` — `fastembed-rs` for the Rust-native adapter (the
/// only option on the standalone libpython-free binary), anything else for a
/// Python library built by the wheel-supplied factory. The rules, and the
/// `trust.allow_embedder` gate they sit behind, live in
/// [`build_embedder_from_manifest`], as does `load:`, which decides whether the
/// model is built here or on its first use.
fn bind_manifest_embedder(
    manifest: Option<&mcp_methods::server::Manifest>,
    py_embedder_factory: Option<&SharedEmbedderFactory>,
    graph_state: &GraphState,
) -> Result<()> {
    if let Some(m) = manifest {
        if let Some(embedder) = build_embedder_from_manifest(m, py_embedder_factory)? {
            graph_state
                .bind_embedder(embedder)
                .context("graph.set_embedder_native failed")?;
        }
    }
    Ok(())
}

/// The `extensions:` knobs the rest of boot is wired from.
struct BootExtensions {
    recipe_catalog: recipe_queries::RecipeCatalog,
    /// Whether the *manifest* declared a non-empty catalog, carried separately
    /// because `recipe_catalog` is about to be merged with layers the operator
    /// did not write. Only this answer arms the allowlist's recipe-route
    /// refusal — see [`apply_tool_allowlist`].
    manifest_recipes_declared: bool,
    tools_allow: Option<Vec<String>>,
    mutations_enabled: bool,
    write_scope: Option<Vec<String>>,
    parallel: bool,
}

/// Settle every `extensions:` knob before the first route, lease or listener
/// exists, so a malformed value fails the boot instead of leaving a running
/// server half-configured.
///
/// The sequence is load-bearing rather than tidy: the unknown-key warning goes
/// out before anything acts on the block, and `mutations_enabled` is decided
/// ahead of [`boot_write_scope`], whose warning asks whether the pin has
/// anything left to apply to.
fn boot_extensions(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
    read_only_pin: bool,
) -> Result<BootExtensions> {
    let unknown_keys = unknown_extension_keys(manifest);
    if !unknown_keys.is_empty() {
        tracing::warn!(
            unknown = unknown_keys.join(","),
            known = KNOWN_EXTENSION_KEYS.join(","),
            "manifest extensions: key(s) this server does not read — check the spelling, or \
             ignore this if a skill's `applies_when: extension_enabled:` names them"
        );
    }
    let recipe_catalog = boot_recipe_catalog(manifest)?;
    let manifest_recipes_declared = !recipe_catalog.is_empty();
    if boot_graph_watch(manifest)?.is_some() {
        tracing::warn!(
            "extensions.graph_watch is retired — a --graph server now refreshes automatically \
             on every tool call; remove the key"
        );
    }
    let tools_allow = boot_tools_allow(manifest)?;
    // Carried as a `bool` from here on: the lease policy, the builtins and the
    // code-tool gate all read this one answer rather than re-deriving it.
    let mutations_enabled = boot_mutations_enabled(manifest, cli, read_only_pin)?;
    let write_scope = boot_write_scope(manifest, cli, mutations_enabled)?;
    let parallel = boot_parallel(manifest, cli)?;
    if parallel {
        let (width, overridden) = query_pool_width();
        tracing::info!(
            pool_threads = width,
            width_source = if overridden {
                "KGLITE_QUERY_THREADS"
            } else {
                "available_parallelism"
            },
            "parallel runtime enabled for reads (--parallel / extensions.parallel); \
             mutations stay sequential"
        );
    }
    Ok(BootExtensions {
        recipe_catalog,
        manifest_recipes_declared,
        tools_allow,
        mutations_enabled,
        write_scope,
        parallel,
    })
}

/// Everything boot settles before the MCP server exists.
///
/// The seam is the server itself: every step here reads argv, the manifest or
/// the graph file and produces state the tool surface is then assembled from,
/// and none of it touches [`McpServer`]. Returned as a struct for the same
/// reason [`BootExtensions`] is one — the set is the boot's output, and a
/// twelve-value tuple names nothing.
struct BootedGraph {
    mode: Mode,
    manifest: Option<mcp_methods::server::Manifest>,
    options: ServerOptions,
    graph_state: GraphState,
    /// The manifest catalogue with the graph's own, and the embedding binary's
    /// under that, merged beneath it.
    recipe_catalog: Arc<recipe_queries::RecipeCatalog>,
    recipe_catalog_hint: Option<recipe_queries::CatalogHint>,
    /// See [`BootExtensions::manifest_recipes_declared`].
    manifest_recipes_declared: bool,
    graph_recipes: recipe_queries::GraphRecipeStats,
    producer_recipes: recipe_queries::ProducerRecipeStats,
    source_root_status: Option<SourceRootStatus>,
    env_file_loaded: Option<PathBuf>,
    tools_allow: Option<Vec<String>>,
    mutations_enabled: bool,
    write_scope: Option<Vec<String>>,
    /// `Some` only in `--vault` mode: where the first-party producer leaves
    /// the report `rebuild_graph` renders.
    vault_report: Option<crate::vault::VaultReportSlot>,
}

/// Everything boot settles from argv, the manifest and the environment before
/// it opens a graph, binds an embedder, listens or watches. `--reload-preflight`
/// runs exactly this and nothing after it, so a build that passes it has parsed
/// every input the real boot parses first.
struct BootPrefix {
    mode: Mode,
    manifest: Option<mcp_methods::server::Manifest>,
    extensions: BootExtensions,
    env_file_loaded: Option<PathBuf>,
    options: ServerOptions,
    value_codecs: Option<Arc<Vec<kglite::api::cypher::ValueCodec>>>,
    valid_time_default: Option<kglite::api::temporal::ValidTimeDefault>,
    embeddings_policy: Option<kglite::api::io::MapPolicy>,
    columns_policy: Option<kglite::api::io::MapPolicy>,
    topology_policy: Option<kglite::api::io::MapPolicy>,
    write_budget: Option<kglite::api::io::WriteBudgetSetting>,
    ontology: Option<BoundOntology>,
}

#[cfg(test)]
thread_local! {
    /// How many times this thread entered [`boot_prefix`].
    static PREFIX_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn boot_prefix(
    cli: &Cli,
    read_only_pin: bool,
    identity: &crate::self_update::Identity,
) -> Result<BootPrefix> {
    #[cfg(test)]
    PREFIX_RUNS.with(|runs| runs.set(runs.get() + 1));
    let mode = pick_mode(cli);
    validate_mode_paths(&mode, cli)?;

    let manifest = load_manifest(cli, &mode).context("manifest load failed")?;
    let extensions = boot_extensions(manifest.as_ref(), cli, read_only_pin)?;

    // Manifest `workspace.kind: local` wins over CLI flags — promote before
    // mode-specific binding so the rest of boot sees `Mode::LocalWorkspace`.
    let mode = promote_local_workspace(mode, manifest.as_ref())?;

    // Load `.env` before anything reads env vars (notably the GitHub tools'
    // `GITHUB_TOKEN` auth check). The walk-up start point matches the framework
    // binary's choice in `mcp-server`'s own main: the mode's directory for
    // source-aware modes, cwd for bare. An explicit manifest `env_file:`
    // overrides walk-up. Returns the path actually loaded, so the boot summary
    // can name it.
    let env_start_dir = resolve_env_start_dir(&mode);
    let env_file_loaded = load_env_for_mode(manifest.as_ref(), &env_start_dir)
        .context("manifest env_file load failed")?;

    let mut options =
        ServerOptions::from_manifest(manifest.as_ref(), &fallback_name(&mode, identity));
    if cli.name.is_some() {
        options.name = cli.name.clone();
    }
    // Fold the lazy-tool-discovery steer into workspace-mode instructions so
    // code-mode / tool-search clients get the "search the registry for cypher"
    // guidance by default, without every deployment copy-pasting it.
    let options = apply_discovery_steer(&mode, options);

    let value_codecs = boot_value_codecs(manifest.as_ref())?;
    let valid_time_default = boot_valid_time_default(manifest.as_ref(), cli)?;
    let embeddings_policy = boot_embeddings_policy(manifest.as_ref(), cli)?;
    let columns_policy = boot_columns_policy(manifest.as_ref(), cli)?;
    let topology_policy = boot_topology_policy(manifest.as_ref(), cli)?;
    let write_budget = boot_write_budget(manifest.as_ref(), cli)?;
    let ontology = boot_ontology(manifest.as_ref(), &manifest_base_dir(manifest.as_ref()))?;
    Ok(BootPrefix {
        mode,
        manifest,
        extensions,
        env_file_loaded,
        options,
        value_codecs,
        valid_time_default,
        embeddings_policy,
        columns_policy,
        topology_policy,
        write_budget,
        ontology,
    })
}

/// `--reload-preflight`: the exit status says whether this build can boot with
/// these flags as far as the side-effect-free prefix can tell.
pub(crate) fn run_reload_preflight(cli: &Cli, read_only_pin: bool) -> Result<()> {
    boot_prefix(cli, read_only_pin, &crate::self_update::Identity::default()).map(|_| ())
}

fn boot_graph(
    cli: &Cli,
    workspace_graph: Option<WorkspaceGraphHooks>,
    producer_recipes: Option<recipe_queries::RecipeCatalog>,
    read_only_pin: bool,
    identity: &crate::self_update::Identity,
    py_embedder_factory: Option<&SharedEmbedderFactory>,
) -> Result<BootedGraph> {
    init_tracing();
    let BootPrefix {
        mode,
        manifest,
        extensions:
            BootExtensions {
                recipe_catalog,
                manifest_recipes_declared,
                tools_allow,
                mutations_enabled,
                write_scope,
                parallel,
            },
        env_file_loaded,
        options,
        value_codecs,
        valid_time_default,
        embeddings_policy,
        columns_policy,
        topology_policy,
        write_budget,
        ontology,
    } = boot_prefix(cli, read_only_pin, identity)?;

    // `--vault`'s producer is this binary's own (decision D5): built here and
    // installed on the state directly, not injected through
    // `ServerExtensions`, because the closure has to hold state the public
    // request type does not carry — the previous graph, for the embedding
    // carry, and the embedder slot. Refuse the collision rather than letting
    // whichever write lands last decide what the server is.
    let graph_state = GraphState::new(workspace_graph_mode(&mode))
        .with_value_codecs(value_codecs)
        .with_parallel(parallel)
        .with_valid_time_default(valid_time_default)
        .with_embeddings_policy(embeddings_policy)
        .with_columns_policy(columns_policy)
        .with_topology_policy(topology_policy)
        .with_write_budget(write_budget)
        // Declared here rather than from `builtins` (built below, after the
        // csv_http boot it needs) because `bind_mode` performs the boot open
        // a few lines down — a policy set later would arrive after the lease
        // decision it governs. Both read the same `owns_graph_file`.
        .with_writer_lease_policy(writer_lease_policy(manifest.as_ref(), mutations_enabled))
        .with_lease_label(Some(boot_lease_label(cli)));

    // The state's own embedder slot is what the vault producer reads, at build
    // time — so a rebuild uses whatever the manifest bound, and the boot build
    // sees it too (bound just below, before `bind_mode`).
    let (workspace_graph, vault_report) = crate::vault::vault_producer(
        &mode,
        workspace_graph,
        &graph_state.embedder,
        crate::cli::vault_cache_policy(cli),
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    let graph_state = graph_state.with_workspace_graph(workspace_graph.map(Arc::new));

    // Bound before `bind_mode`, whose boot open publishes the first graph —
    // the ontology rides the same pre-publication seam as the embedder.
    if let Some(bound) = ontology {
        graph_state.bind_ontology(bound);
    }
    // And the embedder with it, for the same reason plus one more: in vault
    // mode `bind_mode`'s boot build is where `.kglite/vault.yaml`'s `embed:`
    // targets are computed, and an embedder bound after it would leave the
    // booted graph vectorless until the first rebuild. Every install path
    // re-applies it (`apply_bound_embedder`), so binding earlier changes
    // nothing for the other modes except that a malformed `extensions.embedder`
    // now fails before a graph is opened rather than after.
    bind_manifest_embedder(manifest.as_ref(), py_embedder_factory, &graph_state)?;

    let (options, source_root_status) =
        bind_mode(&mode, cli, manifest.as_ref(), &graph_state, options)?;

    // Only now: `bind_mode` has performed the boot open, so the graph's own
    // catalogue exists to merge between the manifest's and the embedder's.
    // Everything downstream — the routes, the bundled `recipe_queries` skill,
    // the overview hint, the raw-stdio route pointers — reads the merged
    // catalogue, so a graph or a producer that carries recipes serves them
    // whether or not the manifest declared any. The one deliberate exception
    // is the allowlist's recipe refusal, which reads `manifest_recipes_declared`
    // because only the operator's own catalogue can contradict their list.
    let (recipe_catalog, producer_recipes, graph_recipes) = recipe_queries::merge_recipe_layers(
        &mode,
        &graph_state,
        producer_recipes.unwrap_or_default(),
        recipe_catalog,
    );
    let recipe_catalog = Arc::new(recipe_catalog);
    let recipe_catalog_hint = recipe_queries::catalog_hint(&recipe_catalog);

    let options = apply_result_decorations(options, &graph_state, source_root_status.as_ref());
    Ok(BootedGraph {
        mode,
        manifest,
        options,
        graph_state,
        recipe_catalog,
        recipe_catalog_hint,
        manifest_recipes_declared,
        graph_recipes,
        producer_recipes,
        source_root_status,
        env_file_loaded,
        tools_allow,
        mutations_enabled,
        write_scope,
        vault_report,
    })
}

/// The catalogue dimensions the skill layer gates on — whether a catalogue is
/// served and how large it is. The `recipe.query` names stay with the overview
/// hint; the skills never read them.
fn skills_catalog_summary(
    hint: Option<&recipe_queries::CatalogHint>,
) -> Option<recipe_queries::CatalogSummary> {
    hint.map(|hint| hint.summary)
}

/// Register the tools only one serving mode has: `reload_graph` (graph mode),
/// `rebuild_graph` (vault mode) and the post-activation skill refresh
/// (workspace modes).
fn register_mode_tools(
    server: &mut McpServer,
    mode: &Mode,
    graph_state: &GraphState,
    refresher: &crate::skills::SkillRefresher,
    vault_report: Option<&crate::vault::VaultReportSlot>,
) {
    if matches!(mode, Mode::Graph { .. }) {
        // `reload_graph` is graph-mode-only: it re-reads *the* served file, an
        // identity no other mode has. Registered from here rather than inside
        // `tools::register`, because the mode is not otherwise visible there.
        tools::register_graph_mode_tools(server, graph_state.clone(), refresher.clone());
    }
    if let Mode::Vault { dir } = &mode {
        // `rebuild_graph` is the producer-backed counterpart of
        // `reload_graph`: registered where a directory, not a file, is the
        // source of truth. Vault mode only — `--watch`'s producer is an
        // injected one whose report this binary cannot render.
        tools::register_vault_mode_tools(
            server,
            graph_state.clone(),
            refresher.clone(),
            dir.canonicalize().unwrap_or_else(|_| dir.clone()),
            vault_report.cloned().unwrap_or_default(),
        );
    }
    if matches!(mode, Mode::Workspace { .. } | Mode::LocalWorkspace { .. }) {
        // The workspace counterpart of `reload_graph`'s refresh: the graph a
        // predicate resolves against does not exist until a root is
        // activated, and the framework owns those two tool handlers. Before
        // `register_extension_tools`, whose tail settles the final route
        // names.
        tools::refresh_skills_after_activation(server, refresher.clone());
    }
}

pub(crate) async fn run_async(
    cli: Cli,
    py_embedder_factory: Option<PyEmbedderFactory>,
    extensions: ServerExtensions,
    launch_args: Vec<std::ffi::OsString>,
) -> Result<()> {
    let ServerExtensions {
        workspace_graph,
        domain_tools,
        read_only: read_only_pin,
        producer_skills,
        producer_recipes,
        identity,
    } = extensions;
    let shared_embedder_factory = crate::embedder::share_factory(py_embedder_factory);
    let BootedGraph {
        mode,
        manifest,
        options,
        graph_state,
        recipe_catalog,
        recipe_catalog_hint,
        manifest_recipes_declared,
        graph_recipes,
        producer_recipes: producer_recipe_stats,
        source_root_status,
        env_file_loaded,
        tools_allow,
        mutations_enabled,
        write_scope,
        vault_report,
    } = boot_graph(
        &cli,
        workspace_graph,
        producer_recipes,
        read_only_pin,
        &identity,
        shared_embedder_factory.as_ref(),
    )?;

    // Snapshot the dynamic source-roots provider before `options` moves into
    // the McpServer. `read_code_source` queries it on every call, so
    // workspace-mode active-repo swaps immediately re-target file resolution.
    let source_roots_provider = options.source_roots.clone();

    let manifest_base = manifest_base_dir(manifest.as_ref());
    let csv_http = Arc::new(boot_csv_http(manifest.as_ref(), &manifest_base).await?);
    let builtins = boot_builtins(
        manifest.as_ref(),
        mutations_enabled,
        &csv_http,
        &manifest_base,
        write_scope,
    );

    // Built before the server so the stale-build check is part of its options;
    // `register_kglite_tools` below receives the same refresher.
    let skill_refresher = crate::skills::SkillRefresher::default();
    let reload = crate::self_update::ReloadParts {
        args: launch_args,
        graph_state: graph_state.clone(),
        refresher: skill_refresher.clone(),
        carry_graph: matches!(mode, Mode::Graph { .. }),
    };
    let options = match crate::self_update::build_self_check(&identity, Some(reload)) {
        Some(check) => options.with_self_check(check),
        None => options,
    };
    let mut server = apply_response_preview(McpServer::new(options));
    // `repo_management` needs no gate here since mcp-methods 0.4.7:
    // `McpServer::new` registers it only for `kind: github` workspaces, so
    // local-workspace (and graph/bare) servers never carry the route. The
    // manifest-override consequence of that removal is handled where the
    // overrides are applied — see the `repo_management` pass in
    // `apply_bundled_tool_overrides`.
    //
    // Read before `builtins` moves into `register_kglite_tools` below.
    let gate_code_tools = code_tools_are_dead(&mode, &builtins, &graph_state);
    // Built here and filled by `install_skills` below: the overview route
    // closes over the decorations long before the skill set is known.
    let skills_index = tools::SkillsIndexSlot::default();
    // Same late-fill shape as `skills_index`, for the same reason: the swap
    // handlers below are registered long before `install_skills` can hand them
    // anything to rebuild.
    let peer_slot = tools::PeerSlot::default();
    register_kglite_tools(
        &mut server,
        KgliteToolParams {
            graph_state: &graph_state,
            manifest: manifest.as_ref(),
            builtins,
            recipe_catalog_hint: recipe_catalog_hint.clone(),
            csv_http: csv_http.clone(),
            source_roots_provider,
            skills_index: skills_index.clone(),
            skill_refresher: skill_refresher.clone(),
        },
    )?;
    register_mode_tools(
        &mut server,
        &mode,
        &graph_state,
        &skill_refresher,
        vault_report.as_ref(),
    );
    if gate_code_tools {
        disable_code_tools(&mut server);
    }
    // Register YAML Cypher tools, then downstream domain routes. Keeping this
    // before skill finalisation makes every route visible to predicates.
    let (recipe_routes, named_recipe_tools) =
        recipe_route_plan(manifest.as_ref(), &recipe_catalog)?;
    register_extension_tools(
        &mut server,
        &graph_state,
        manifest.as_ref(),
        &csv_http,
        recipe_catalog,
        &recipe_routes,
        domain_tools,
    )?;

    // Last, and only here: the allowlist closes the surface once *every* route
    // source has registered and `apply_bundled_tool_overrides` (at the tail of
    // `register_extension_tools`) has settled the final names. Applying it any
    // earlier would let a later registration re-widen the surface, and would
    // match against pre-rename names the agent never sees. Before skills, so a
    // `tool_registered:` predicate sees the closed surface.
    if let Some(allow) = tools_allow.as_deref() {
        apply_tool_allowlist(&mut server, allow, manifest_recipes_declared)
            .context("extensions.tools_allow could not be applied")?;
    }

    let _watch_handle = spawn_mode_watcher(&mode, &graph_state)?;

    // The skill refresh for lazy swaps (watcher rebuild, `--graph` re-read).
    // Armed after `boot_skills` below would be too late for the first rebuild
    // and too early here only in the sense that the refresher is still unarmed
    // — which `SkillRefresher::refresh` treats as the no-op it is.
    {
        let refresher = skill_refresher.clone();
        graph_state.set_after_rebuild(Arc::new(move || refresher.refresh()));
    }

    let skill_stats = crate::skills::boot_skills(
        &mut server,
        crate::skills::SkillBootParams {
            producer_records: producer_skills,
            manifest: manifest.as_ref(),
            mode: &mode,
            graph_state: &graph_state,
            recipe_catalog_summary: skills_catalog_summary(recipe_catalog_hint.as_ref()),
            skills_index: &skills_index,
            refresher: &skill_refresher,
            peer: &peer_slot,
        },
    )?;

    print_boot_summary(
        &mode,
        manifest.as_ref(),
        &graph_state,
        env_file_loaded.as_deref(),
        &csv_http,
        source_root_status.as_ref(),
        crate::boot::ServedLayers {
            producer_skills: &skill_stats.producer,
            producer_recipes: &producer_recipe_stats,
            graph_skills: &skill_stats.graph,
            graph_recipes: &graph_recipes,
        },
    );

    let query_routes = crate::raw_query_routes::route_pointers(
        manifest.as_ref(),
        recipe_catalog_hint.is_some(),
        &named_recipe_tools,
    );
    let announce_server = server.clone();
    let service = server
        .serve_with_handover(
            mcp_methods::server::stdio::StdioTransport::stdio().with_recover(move |raw| {
                crate::query_recover::recover_query_message(raw, &query_routes)
            }),
        )
        .await
        .context("failed to start MCP service over stdio")?;
    // The one handle on the connected client anything downstream can reach:
    // dynamically registered tool handlers get no `RequestContext`, so a
    // post-swap `tools/list_changed` has to be announced through this. The
    // server clone shares the `subscriptions/listen` registry the stateless
    // (2026-07-28) clients are announced to.
    *tools::write_lock(&peer_slot) = Some(tools::ListAnnouncer {
        server: announce_server,
        peer: service.peer().clone(),
    });
    service.waiting().await?;
    Ok(())
}

/// Every boot knob read here must survive the whole chain: spelled in a YAML
/// file, accepted by mcp-methods' loader (which validates nothing inside
/// `extensions:`), and read back here as the type the wiring expects. Loading
/// through the real loader rather than constructing a `Manifest` is deliberate
/// — a struct literal would skip the two steps most likely to break.
#[cfg(test)]
mod boot_manifest_tests {
    use super::*;

    #[test]
    fn reload_preflight_and_the_real_boot_run_the_same_prefix() {
        use clap::Parser;
        let tmp = tempfile::tempdir().expect("tempdir");
        let manifest = manifest_with(tmp.path(), "name: prefix\n");
        let cli = Cli::try_parse_from([
            std::ffi::OsString::from("kglite-mcp-server"),
            "--mcp-config".into(),
            manifest.yaml_path.clone().into_os_string(),
        ])
        .expect("flags parse");
        let runs = || PREFIX_RUNS.with(|runs| runs.get());
        let before = runs();
        run_reload_preflight(&cli, false).expect("preflight passes");
        assert_eq!(runs(), before + 1, "--reload-preflight runs boot_prefix");
        boot_graph(
            &cli,
            None,
            None,
            false,
            &crate::self_update::Identity::default(),
            None,
        )
        .expect("boot passes");
        assert_eq!(runs(), before + 2, "boot_graph runs boot_prefix");
    }

    #[test]
    fn reload_preflight_rejects_what_the_real_boot_rejects() {
        use clap::Parser;
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("bad.yaml");
        std::fs::write(&path, "name: [unterminated\n").expect("write");
        let cli = Cli::try_parse_from([
            std::ffi::OsString::from("kglite-mcp-server"),
            "--mcp-config".into(),
            path.into_os_string(),
        ])
        .expect("flags parse");
        assert!(run_reload_preflight(&cli, false).is_err());
    }

    fn manifest_with(dir: &Path, body: &str) -> mcp_methods::server::Manifest {
        let path = dir.join("graph_watch_mcp.yaml");
        std::fs::write(&path, body).expect("write manifest");
        mcp_methods::server::load_manifest(&path).expect("manifest loads")
    }

    #[test]
    fn recipe_numeric_bounds_accept_reachable_yaml_signed_and_finite_values() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let manifest = manifest_with(
            tmp.path(),
            "extensions:\n\
             \x20 cypher_recipes:\n\
             \x20   numeric:\n\
             \x20     description: Numeric.\n\
             \x20     queries:\n\
             \x20       echo:\n\
             \x20         description: Echo.\n\
             \x20         parameters:\n\
             \x20           type: object\n\
             \x20           properties:\n\
             \x20             value:\n\
             \x20               type: number\n\
             \x20               minimum: -1e300\n\
             \x20               maximum: 9223372036854775807\n\
             \x20           required: [value]\n\
             \x20           additionalProperties: false\n\
             \x20         cypher: RETURN $value AS value\n",
        );
        let catalog = boot_recipe_catalog(Some(&manifest)).expect("reachable bounds compile");
        assert_eq!(catalog.discovery_summary().unwrap().query_count, 1);
    }

    /// The retired key is *accepted*, not honoured: a manifest that still
    /// carries it must boot (four MCP clients share one manifest, and an
    /// operator cannot edit it the moment this server upgrades), and the boot
    /// must be able to tell it apart from an absent key so it can say the key
    /// no longer does anything. Both values are declared present, including
    /// `false` — that is the one an author wrote to mean "never refresh", and
    /// it is the one whose meaning this release took away.
    #[test]
    fn a_graph_watch_key_is_accepted_and_reported_as_retired() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            boot_graph_watch(None).expect("no manifest parses"),
            None,
            "no manifest carries no key to warn about"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert_eq!(
            boot_graph_watch(Some(&bare)).expect("bare manifest parses"),
            None,
            "an absent key must not produce a retirement warning"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  graph_watch: false\n");
        assert_eq!(
            boot_graph_watch(Some(&off)).expect("explicit false parses"),
            Some(false),
            "`false` no longer disables anything, so its author must be told"
        );
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  graph_watch: true\n");
        assert_eq!(
            boot_graph_watch(Some(&on)).expect("explicit true parses"),
            Some(true),
            "and `true` is now what every --graph server does anyway"
        );
    }

    /// The name a peer reads instead of "another process". Precedence is
    /// `--lease-label` > `KGLITE_LEASE_LABEL` > parent process name; only the
    /// flag half is asserted here, because the other two read process state a
    /// unit test cannot set without racing every other test in the binary.
    #[test]
    fn an_explicit_lease_label_wins_and_a_blank_one_falls_through() {
        let named = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--lease-label",
            "  Claude Desktop  ",
        ]);
        assert_eq!(boot_lease_label(&named), "Claude Desktop");
        // A blank flag is not a name. Falling through matters because the
        // fallback is what a peer reads in the owner record instead of
        // "another process".
        let blank = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--lease-label",
            " ",
        ]);
        assert!(!boot_lease_label(&blank).is_empty());
    }

    /// Boot wiring for the writer-lease policy: which deployments own the
    /// served path. The state-level consequences (lease taken or not, disk
    /// directories exempt) are pinned in `tools::tests::lifecycle`; this pins
    /// only the mapping from CLI/manifest to policy.
    #[test]
    fn only_write_enabled_deployments_take_the_writer_lease() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            writer_lease_policy(None, false),
            tools::WriterLeasePolicy::ReadOnly,
            "a read-only --graph server must leave the file lockable by a rebuilder"
        );

        assert_eq!(
            writer_lease_policy(None, true),
            tools::WriterLeasePolicy::Exclusive,
            "a write-enabled server owns the file it serves"
        );

        // `save_graph` without writes enabled: still a writer of that file.
        let saver = manifest_with(tmp.path(), "name: saver\nbuiltins:\n  save_graph: true\n");
        assert_eq!(
            writer_lease_policy(Some(&saver), false),
            tools::WriterLeasePolicy::Exclusive
        );
        let plain = manifest_with(tmp.path(), "name: plain\n");
        assert_eq!(
            writer_lease_policy(Some(&plain), false),
            tools::WriterLeasePolicy::ReadOnly,
            "a manifest that enables nothing must not re-arm the lease"
        );

        // The manifest-writable arm reaches the same policy as the flag, via
        // the one resolution every consumer reads.
        let read_only_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let manifest_writable =
            manifest_with(tmp.path(), "name: mw\nextensions:\n  writable: true\n");
        assert_eq!(
            writer_lease_policy(
                Some(&manifest_writable),
                boot_mutations_enabled(Some(&manifest_writable), &read_only_cli, false)
                    .expect("extensions.writable parses"),
            ),
            tools::WriterLeasePolicy::Exclusive,
            "extensions.writable must reach the lease decision; if this fails the key is \
             parsed and then dropped"
        );
    }

    /// `--writable` and `extensions.writable` are one operator pair, resolved
    /// once. Pinned like [`parallel_reads_the_flag_and_the_manifest`], because
    /// a manifest-only deployment (a wrapper that never controls argv) and a
    /// flag-only one (the bare binary, no manifest) are both real.
    #[test]
    fn mutations_read_the_flag_and_the_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);

        assert!(
            !boot_mutations_enabled(None, &bare_cli, false).expect("no manifest parses"),
            "neither surface set must leave the server read-only"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            !boot_mutations_enabled(Some(&bare), &bare_cli, false).expect("bare manifest parses"),
            "an absent key must leave the server read-only — otherwise the on-arm proves nothing"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  writable: false\n");
        assert!(
            !boot_mutations_enabled(Some(&off), &bare_cli, false).expect("explicit false parses")
        );

        assert!(boot_mutations_enabled(None, &flag_cli, false).expect("flag alone parses"));
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  writable: true\n");
        assert!(
            boot_mutations_enabled(Some(&on), &bare_cli, false).expect("explicit true parses"),
            "extensions.writable: true must reach the mutation gate; if this fails the key \
             is parsed and then dropped"
        );
        assert!(
            boot_mutations_enabled(Some(&on), &flag_cli, false).expect("both parse"),
            "flag + manifest is their OR — neither surface can switch the other off"
        );
        assert!(
            boot_mutations_enabled(Some(&off), &flag_cli, false).expect("both parse"),
            "an explicit manifest `false` does not veto the flag; the pair is an OR, and a \
             flag the operator typed on this launch is the more specific statement"
        );

        // `builtins.save_graph` is the *other* predicate and must not leak
        // into this one — that fusion is the bug this split undoes.
        let saver = manifest_with(tmp.path(), "name: saver\nbuiltins:\n  save_graph: true\n");
        assert!(
            !boot_mutations_enabled(Some(&saver), &bare_cli, false)
                .expect("save_graph manifest parses"),
            "builtins.save_graph: true registers save_graph; it does not open cypher_query \
             to mutations"
        );
    }

    /// The embedder's pin beats both operator surfaces.
    ///
    /// sonagram embeds this server with a closed `tools_allow` and owns argv
    /// but not the manifest: an operator editing the sibling manifest could
    /// set `extensions.writable: true` and open writes against a graph their
    /// next build regenerates. The pin is the contract that makes that
    /// impossible without the embedder re-implementing the resolution — the
    /// text-grep workaround it replaces missed exactly this.
    #[test]
    fn the_read_only_pin_beats_every_write_opt_in() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  writable: true\n");

        assert!(
            !boot_mutations_enabled(None, &flag_cli, true).expect("flag alone parses"),
            "--writable must not open a pinned server"
        );
        assert!(
            !boot_mutations_enabled(Some(&on), &bare_cli, true).expect("manifest key parses"),
            "extensions.writable: true must not open a pinned server — this is the exact \
             case the embedder cannot see from argv"
        );
        assert!(
            !boot_mutations_enabled(Some(&on), &flag_cli, true).expect("both parse"),
            "both surfaces together must not open a pinned server"
        );
        assert!(
            !boot_mutations_enabled(None, &bare_cli, true).expect("neither parses"),
            "and neither leaves it read-only, as it already was"
        );

        // Non-vacuity: the same three inputs without the pin must all open.
        // Without this, a `boot_mutations_enabled` returning `false`
        // unconditionally would satisfy every assertion above.
        assert!(boot_mutations_enabled(None, &flag_cli, false).expect("flag alone parses"));
        assert!(boot_mutations_enabled(Some(&on), &bare_cli, false).expect("manifest parses"));
        assert!(boot_mutations_enabled(Some(&on), &flag_cli, false).expect("both parse"));
    }

    /// The two predicates as the router sees them. `writable` gates
    /// `cypher_query` mutations and the lifecycle tools; `save_graph` gates
    /// only the save route, and is the union of the two — so the
    /// `save_graph`-only manifest is the one arm where they differ.
    #[test]
    fn builtins_separate_the_save_route_from_the_mutation_gate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        let csv_off = csv_http::CsvHttpState::Off;

        let resolve = |manifest: Option<&mcp_methods::server::Manifest>, cli: &Cli| {
            let mutations_enabled =
                boot_mutations_enabled(manifest, cli, false).expect("manifest writable parses");
            let builtins = boot_builtins(manifest, mutations_enabled, &csv_off, tmp.path(), None);
            (builtins.writable, builtins.save_graph)
        };

        assert_eq!(
            resolve(None, &flag_cli),
            (true, true),
            "--writable implies save_graph: an agent that mutates needs somewhere to persist"
        );
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  writable: true\n");
        assert_eq!(
            resolve(Some(&on), &bare_cli),
            (true, true),
            "extensions.writable is the same statement as the flag, made in the manifest"
        );
        let both = manifest_with(
            tmp.path(),
            "name: both\nextensions:\n  writable: true\nbuiltins:\n  save_graph: true\n",
        );
        assert_eq!(
            resolve(Some(&both), &flag_cli),
            (true, true),
            "every surface saying yes is still one yes — the union, not a conflict"
        );
        let saver = manifest_with(tmp.path(), "name: saver\nbuiltins:\n  save_graph: true\n");
        assert_eq!(
            resolve(Some(&saver), &bare_cli),
            (false, true),
            "builtins.save_graph: true alone registers save_graph and leaves cypher_query \
             read-only"
        );
        let plain = manifest_with(tmp.path(), "name: plain\n");
        assert_eq!(
            resolve(Some(&plain), &bare_cli),
            (false, false),
            "a manifest that enables nothing must expose neither"
        );
    }

    #[test]
    fn a_malformed_writable_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);

        let word = manifest_with(
            tmp.path(),
            "name: word\nextensions:\n  writable: yes-please\n",
        );
        let error = boot_mutations_enabled(Some(&word), &cli, false)
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  writable:\n");
        assert!(
            boot_mutations_enabled(Some(&null), &cli, false).is_err(),
            "an empty (null) value is a typo, not `false`"
        );

        let mapping = manifest_with(
            tmp.path(),
            "name: mapping\nextensions:\n  writable:\n    enabled: true\n",
        );
        assert!(
            boot_mutations_enabled(Some(&mapping), &cli, false).is_err(),
            "a mapping is a typo, not a truthy value"
        );
    }

    /// A misspelled key is the one failure mode `extensions.writable` adds
    /// that a mistyped *value* does not cover: it parses, it is stored, and
    /// nothing reads it. Every key a `boot_*` reader consumes must be absent
    /// from this list, or the warning cries wolf on a working manifest.
    #[test]
    fn an_unreadable_extensions_key_is_reported_and_every_read_key_is_not() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(unknown_extension_keys(None).is_empty());
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(unknown_extension_keys(Some(&bare)).is_empty());

        let typo = manifest_with(tmp.path(), "name: typo\nextensions:\n  writeable: true\n");
        assert_eq!(
            unknown_extension_keys(Some(&typo)),
            vec!["writeable".to_string()],
            "the near-miss spelling of the write opt-in must be named at boot; unwarned, it \
             is a read-only server the operator believes is writable"
        );

        // One entry per key this crate actually reads, so a key that stops
        // being read (or a new reader whose key nobody registered) shows up
        // here rather than as a boot warning on a manifest that works.
        let all_keys = manifest_with(
            tmp.path(),
            "name: all\nextensions:\n  cypher_recipes: {}\n  recipe_catalog: {}\n  \
             recipe_tools: true\n  value_codecs: []\n  ontology: {}\n  graph_watch: true\n  parallel: true\n  tools_allow: []\n  \
             write_scope: []\n  csv_http_server: false\n  embedder: {}\n  writable: true\n  \
             fetch_images: {}\n  valid_time: {}\n  embeddings: {}\n  columns: {}\n  topology: {}\n  write_budget: {}\n",
        );
        assert!(
            unknown_extension_keys(Some(&all_keys)).is_empty(),
            "every key with a boot_* reader must be in KNOWN_EXTENSION_KEYS: {:?}",
            unknown_extension_keys(Some(&all_keys))
        );
        assert_eq!(
            all_keys.extensions.len(),
            KNOWN_EXTENSION_KEYS.len(),
            "this manifest must exercise every known key, or the list can grow stale entries \
             unnoticed"
        );
    }

    /// The absent-key arm is what makes the present-key arm mean anything.
    #[test]
    fn recipe_catalog_budgets_default_and_read_the_manifest_block() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let defaults = recipe_queries::CatalogBudgets::default();
        assert_eq!(
            boot_recipe_catalog_budgets(None).expect("no manifest parses"),
            defaults
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert_eq!(
            boot_recipe_catalog_budgets(Some(&bare)).expect("bare manifest parses"),
            defaults,
            "an absent key must leave both shipped budgets in place"
        );

        let set = manifest_with(
            tmp.path(),
            "name: set\nextensions:\n  recipe_catalog:\n    block_budget: 32000\n    \
             description_budget: 250\n",
        );
        let budgets = boot_recipe_catalog_budgets(Some(&set)).expect("block parses");
        assert_eq!((budgets.block, budgets.description), (32_000, 250));

        let typo = manifest_with(
            tmp.path(),
            "name: typo\nextensions:\n  recipe_catalog:\n    descripton_budget: 250\n",
        );
        assert!(
            boot_recipe_catalog_budgets(Some(&typo)).is_err(),
            "a misspelled budget must fail the boot, not silently keep the default"
        );
    }

    /// Named recipe tools are on unless the operator says no — the same
    /// posture as producer skills — and a non-boolean is a boot error.
    #[test]
    fn recipe_tools_default_on_and_read_the_manifest_switch() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(boot_recipe_tools_enabled(None).expect("no manifest parses"));
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(boot_recipe_tools_enabled(Some(&bare)).expect("bare manifest parses"));

        let off = manifest_with(
            tmp.path(),
            "name: off\nextensions:\n  recipe_tools: false\n",
        );
        assert!(!boot_recipe_tools_enabled(Some(&off)).expect("switch parses"));
        assert!(
            !boot_recipe_routes(Some(&off))
                .expect("options parse")
                .named_tools
        );

        let bad = manifest_with(
            tmp.path(),
            "name: bad\nextensions:\n  recipe_tools: maybe\n",
        );
        assert!(
            boot_recipe_tools_enabled(Some(&bad)).is_err(),
            "a non-boolean must fail the boot rather than read as off"
        );
    }

    /// The absent-key arm is what makes the present-key arm mean anything.
    #[test]
    fn tools_allow_defaults_to_absent_and_reads_the_manifest_list() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(boot_tools_allow(None)
            .expect("no manifest parses")
            .is_none());
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            boot_tools_allow(Some(&bare))
                .expect("bare manifest parses")
                .is_none(),
            "an absent key must leave the surface open"
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  tools_allow:\n    - cypher_query\n    - graph_overview\n    - ping\n",
        );
        assert_eq!(
            boot_tools_allow(Some(&listed)).expect("list parses"),
            Some(vec![
                "cypher_query".to_string(),
                "graph_overview".to_string(),
                "ping".to_string(),
            ]),
            "extensions.tools_allow must reach the router wiring; if this fails the key is \
             parsed and then dropped"
        );

        let empty = manifest_with(tmp.path(), "name: empty\nextensions:\n  tools_allow: []\n");
        assert_eq!(
            boot_tools_allow(Some(&empty)).expect("empty list parses"),
            Some(Vec::new()),
            "an explicit empty list closes the surface completely — it is not 'no allowlist'"
        );
    }

    #[test]
    fn a_malformed_tools_allow_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let scalar = manifest_with(
            tmp.path(),
            "name: scalar\nextensions:\n  tools_allow: cypher_query\n",
        );
        let error = boot_tools_allow(Some(&scalar))
            .expect_err("a scalar must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a list"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  tools_allow:\n");
        assert!(
            boot_tools_allow(Some(&null)).is_err(),
            "an empty (null) value is a typo, not an empty list"
        );

        let mistyped = manifest_with(
            tmp.path(),
            "name: mistyped\nextensions:\n  tools_allow:\n    - ping\n    - 7\n",
        );
        let error = boot_tools_allow(Some(&mistyped))
            .expect_err("a non-string element must fail boot")
            .to_string();
        assert!(error.contains("must be tool names"), "{error}");
    }

    #[test]
    fn write_scope_reads_the_flag_the_manifest_and_their_intersection() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        assert!(
            boot_write_scope(None, &bare_cli, true)
                .expect("no pin parses")
                .is_none(),
            "no flag and no manifest must leave writes unpinned"
        );

        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Plan,Task",
        ]);
        assert_eq!(
            boot_write_scope(None, &flag, true).expect("flag parses"),
            Some(vec!["Plan".to_string(), "Task".to_string()])
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  write_scope:\n    - Plan\n    - Task\n",
        );
        assert_eq!(
            boot_write_scope(Some(&listed), &bare_cli, true).expect("list parses"),
            Some(vec!["Plan".to_string(), "Task".to_string()]),
            "extensions.write_scope must reach the tool wiring; if this fails the key is \
             parsed and then dropped"
        );

        // Both surfaces set: neither can widen the other.
        let narrower = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Task,Algorithm",
        ]);
        assert_eq!(
            boot_write_scope(Some(&listed), &narrower, true).expect("both parse"),
            Some(vec!["Task".to_string()]),
            "flag + manifest is their intersection, not either one alone"
        );

        let empty = manifest_with(tmp.path(), "name: empty\nextensions:\n  write_scope: []\n");
        assert_eq!(
            boot_write_scope(Some(&empty), &bare_cli, true).expect("empty list parses"),
            Some(Vec::new()),
            "an explicit empty list pins 'no writes' — it is not 'no pin'"
        );
    }

    #[test]
    fn a_malformed_write_scope_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);

        let scalar = manifest_with(
            tmp.path(),
            "name: scalar\nextensions:\n  write_scope: Task\n",
        );
        let error = boot_write_scope(Some(&scalar), &cli, true)
            .expect_err("a scalar must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a list"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  write_scope:\n");
        assert!(
            boot_write_scope(Some(&null), &cli, true).is_err(),
            "an empty (null) value is a typo, not an empty list"
        );

        let mistyped = manifest_with(
            tmp.path(),
            "name: mistyped\nextensions:\n  write_scope:\n    - Task\n    - 7\n",
        );
        let error = boot_write_scope(Some(&mistyped), &cli, true)
            .expect_err("a non-string element must fail boot")
            .to_string();
        assert!(error.contains("must be node types"), "{error}");
    }

    #[test]
    fn a_non_boolean_graph_watch_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bad = manifest_with(
            tmp.path(),
            "name: bad\nextensions:\n  graph_watch: yes-please\n",
        );
        let error = boot_graph_watch(Some(&bad))
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");
    }

    /// The parallel opt-in is an operator *pair* — `--parallel` and
    /// `extensions.parallel` — so the boot resolution is the OR of the two.
    /// Pinned here because a manifest-only deployment (a wrapper that never
    /// controls argv) and a flag-only one (the bare binary, no manifest) are
    /// both real, and either surface silently losing its half looks exactly
    /// like "the knob does nothing".
    #[test]
    fn parallel_reads_the_flag_and_the_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--parallel"]);

        assert!(
            !boot_parallel(None, &bare_cli).expect("no manifest parses"),
            "neither surface set must leave the engine sequential"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            !boot_parallel(Some(&bare), &bare_cli).expect("bare manifest parses"),
            "an absent key must leave the engine sequential — otherwise the on-arm proves nothing"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  parallel: false\n");
        assert!(!boot_parallel(Some(&off), &bare_cli).expect("explicit false parses"));

        assert!(
            boot_parallel(None, &flag_cli).expect("flag alone parses"),
            "--parallel must reach the execution wiring; if this fails the flag is parsed \
             and then dropped"
        );
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  parallel: true\n");
        assert!(
            boot_parallel(Some(&on), &bare_cli).expect("explicit true parses"),
            "extensions.parallel: true must reach the execution wiring; if this fails the \
             key is parsed and then dropped"
        );
        assert!(
            boot_parallel(Some(&on), &flag_cli).expect("both parse"),
            "flag + manifest is their OR — neither surface can switch the other off"
        );
        assert!(
            boot_parallel(Some(&off), &flag_cli).expect("both parse"),
            "an explicit manifest `false` does not veto the flag; the pair is an OR, and a \
             flag the operator typed on this launch is the more specific statement"
        );
    }

    #[test]
    fn valid_time_default_reads_the_manifest_and_the_flag_wins() {
        use kglite::api::temporal::ValidTimeDefault as D;
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--valid-time-default",
            "2015-06-15",
        ]);
        assert_eq!(boot_valid_time_default(None, &bare).unwrap(), None);
        assert_eq!(
            boot_valid_time_default(None, &flag)
                .unwrap()
                .map(|d| d.to_string()),
            Some("2015-06-15".to_string())
        );
        let all = manifest_with(
            tmp.path(),
            "name: a\nextensions:\n  valid_time:\n    default: all\n",
        );
        assert_eq!(
            boot_valid_time_default(Some(&all), &bare).unwrap(),
            Some(D::All)
        );
        assert_eq!(
            boot_valid_time_default(Some(&all), &flag)
                .unwrap()
                .map(|d| d.to_string()),
            Some("2015-06-15".to_string()),
            "the flag overrides the manifest"
        );
        let empty = manifest_with(tmp.path(), "name: e\nextensions:\n  valid_time: {}\n");
        assert_eq!(boot_valid_time_default(Some(&empty), &bare).unwrap(), None);
        assert!(KNOWN_EXTENSION_KEYS.contains(&"valid_time"));
    }

    #[test]
    fn embeddings_policy_reads_flag_over_manifest_and_defaults_to_none() {
        use kglite::api::io::MapPolicy;
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--embeddings",
            "memory",
        ]);
        assert_eq!(boot_embeddings_policy(None, &bare).unwrap(), None);
        assert_eq!(
            boot_embeddings_policy(None, &flag).unwrap(),
            Some(MapPolicy::Memory)
        );
        let mapped = manifest_with(
            tmp.path(),
            "name: a\nextensions:\n  embeddings:\n    storage: mapped\n",
        );
        assert_eq!(
            boot_embeddings_policy(Some(&mapped), &bare).unwrap(),
            Some(MapPolicy::Mapped)
        );
        assert_eq!(
            boot_embeddings_policy(Some(&mapped), &flag).unwrap(),
            Some(MapPolicy::Memory),
            "the flag overrides the manifest"
        );
        let empty = manifest_with(tmp.path(), "name: e\nextensions:\n  embeddings: {}\n");
        assert_eq!(boot_embeddings_policy(Some(&empty), &bare).unwrap(), None);
        assert!(KNOWN_EXTENSION_KEYS.contains(&"embeddings"));
        assert!(Cli::try_parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--embeddings",
            "always"
        ])
        .is_err());
    }

    #[test]
    fn columns_policy_reads_flag_over_manifest_and_defaults_to_none() {
        use kglite::api::io::MapPolicy;
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--columns",
            "memory",
        ]);
        assert_eq!(boot_columns_policy(None, &bare).unwrap(), None);
        assert_eq!(
            boot_columns_policy(None, &flag).unwrap(),
            Some(MapPolicy::Memory)
        );
        let mapped = manifest_with(
            tmp.path(),
            "name: a\nextensions:\n  columns:\n    storage: mapped\n",
        );
        assert_eq!(
            boot_columns_policy(Some(&mapped), &bare).unwrap(),
            Some(MapPolicy::Mapped)
        );
        assert_eq!(
            boot_columns_policy(Some(&mapped), &flag).unwrap(),
            Some(MapPolicy::Memory),
            "the flag overrides the manifest"
        );
        assert_eq!(
            boot_embeddings_policy(Some(&mapped), &bare).unwrap(),
            None,
            "the two keys are independent"
        );
        assert!(KNOWN_EXTENSION_KEYS.contains(&"columns"));
        for body in ["storage: always", "storage: 7", "stroage: mapped"] {
            let manifest = manifest_with(
                tmp.path(),
                &format!("name: m\nextensions:\n  columns:\n    {body}\n"),
            );
            assert!(
                boot_columns_policy(Some(&manifest), &bare).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn topology_policy_reads_flag_over_manifest_and_defaults_to_none() {
        use kglite::api::io::MapPolicy;
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--topology",
            "memory",
        ]);
        assert_eq!(boot_topology_policy(None, &bare).unwrap(), None);
        assert_eq!(
            boot_topology_policy(None, &flag).unwrap(),
            Some(MapPolicy::Memory)
        );
        let mapped = manifest_with(
            tmp.path(),
            "name: a\nextensions:\n  topology:\n    storage: mapped\n",
        );
        assert_eq!(
            boot_topology_policy(Some(&mapped), &bare).unwrap(),
            Some(MapPolicy::Mapped)
        );
        assert_eq!(
            boot_topology_policy(Some(&mapped), &flag).unwrap(),
            Some(MapPolicy::Memory),
            "the flag overrides the manifest"
        );
        assert_eq!(
            boot_columns_policy(Some(&mapped), &bare).unwrap(),
            None,
            "the keys are independent"
        );
        assert!(KNOWN_EXTENSION_KEYS.contains(&"topology"));
        for body in ["storage: always", "storage: 7", "stroage: mapped"] {
            let manifest = manifest_with(
                tmp.path(),
                &format!("name: m\nextensions:\n  topology:\n    {body}\n"),
            );
            assert!(
                boot_topology_policy(Some(&manifest), &bare).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn write_budget_reads_flag_over_manifest_and_defaults_to_none() {
        use kglite::api::io::WriteBudgetSetting;
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--write-budget",
            "unlimited",
        ]);
        assert_eq!(boot_write_budget(None, &bare).unwrap(), None);
        assert_eq!(
            boot_write_budget(None, &flag).unwrap(),
            Some(WriteBudgetSetting::Unlimited)
        );
        let sized = manifest_with(
            tmp.path(),
            "name: a\nextensions:\n  write_budget:\n    limit: 2GiB\n",
        );
        assert_eq!(
            boot_write_budget(Some(&sized), &bare).unwrap(),
            Some(WriteBudgetSetting::Bytes(2 << 30))
        );
        assert_eq!(
            boot_write_budget(Some(&sized), &flag).unwrap(),
            Some(WriteBudgetSetting::Unlimited),
            "the flag overrides the manifest"
        );
        assert!(KNOWN_EXTENSION_KEYS.contains(&"write_budget"));
        for body in ["limit: lots", "limit: [1]", "size: 5"] {
            let manifest = manifest_with(
                tmp.path(),
                &format!("name: m\nextensions:\n  write_budget:\n    {body}\n"),
            );
            assert!(boot_write_budget(Some(&manifest), &bare).is_err(), "{body}");
        }
        assert!(Cli::try_parse_from(["kglite-mcp-server", "--write-budget", "huge"]).is_err());
    }

    #[test]
    fn a_malformed_embeddings_block_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        for body in ["storage: always", "storage: 7", "stroage: mapped"] {
            let manifest = manifest_with(
                tmp.path(),
                &format!("name: m\nextensions:\n  embeddings:\n    {body}\n"),
            );
            assert!(
                boot_embeddings_policy(Some(&manifest), &cli).is_err(),
                "{body}"
            );
        }
        let scalar = manifest_with(tmp.path(), "name: s\nextensions:\n  embeddings: mapped\n");
        assert!(boot_embeddings_policy(Some(&scalar), &cli).is_err());
    }

    #[test]
    fn a_malformed_valid_time_block_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        for body in ["default: yesterday", "default: 7", "dfault: all"] {
            let manifest = manifest_with(
                tmp.path(),
                &format!("name: m\nextensions:\n  valid_time:\n    {body}\n"),
            );
            assert!(
                boot_valid_time_default(Some(&manifest), &cli).is_err(),
                "{body}"
            );
        }
        let scalar = manifest_with(tmp.path(), "name: s\nextensions:\n  valid_time: all\n");
        assert!(boot_valid_time_default(Some(&scalar), &cli).is_err());
    }

    #[test]
    fn a_malformed_parallel_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);

        let word = manifest_with(
            tmp.path(),
            "name: word\nextensions:\n  parallel: yes-please\n",
        );
        let error = boot_parallel(Some(&word), &cli)
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  parallel:\n");
        assert!(
            boot_parallel(Some(&null), &cli).is_err(),
            "an empty (null) value is a typo, not `false`"
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  parallel:\n    - true\n",
        );
        assert!(
            boot_parallel(Some(&listed), &cli).is_err(),
            "a list is a typo, not a truthy value"
        );
    }
}
